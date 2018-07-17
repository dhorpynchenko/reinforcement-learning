import os
from abc import abstractmethod, ABCMeta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import trange
from gym.spaces import Discrete, Box

import utils
from environment import Environment


class Agent(metaclass=ABCMeta):

    def __init__(self, name, academy, action_space):
        self.action_space = action_space
        self.name = name
        self.academy = academy

    @abstractmethod
    def save(self, file):
        pass

    @abstractmethod
    def restore(self, file) -> bool:
        pass

    @abstractmethod
    def action(self, observation):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def training_action(self, observation, progress):
        pass

    @abstractmethod
    def teach_single(self, prev_observation, action, curr_observation, reward) -> (bool, float):
        pass

    @abstractmethod
    def teach_survey(self, data) -> (bool, float):
        """
        Teach agent with data per episode
        :param data: list with episode data.
        data[0], data[2] - observation before and after action; data[1] - action, data[3] - reward
        :return True if algorithm converged, False otherwise; cumulative loss
        """
        pass


class Academy:

    def __init__(self, save_folder="./AcademySave"):
        self.save_folder = save_folder

    def _get_agent_save_file_name(self, agent: Agent):
        return agent.name

    def _get_env_save_folder(self, env: Environment):
        return os.path.join(self.save_folder, env.name)

    def random_agent(self, environment: Environment) -> Agent:
        return Academy.RandomAgent(self, environment.action_space())

    def table_method_agent(self, environment: Environment) -> Agent:
        return Academy.TableMethodAgent(self, environment.observation_space(), environment.action_space())

    def feed_forward_network_agent(self, environment: Environment) -> Agent:
        return Academy.FeedForwardNetworkAgent(self, environment.observation_space(), environment.action_space())

    def save_agent_settings(self, agent: Agent, environment: Environment):
        env_save_folder = self._get_env_save_folder(environment)
        os.makedirs(env_save_folder, 0, True)
        agent.save(os.path.join(env_save_folder, self._get_agent_save_file_name(agent)))

    def restore_agent_settings(self, agent: Agent, env: Environment):
        return agent.restore(os.path.join(self._get_env_save_folder(env), self._get_agent_save_file_name(agent)))

    class RandomAgent(Agent):
        """
        Acts randomly
        """

        def __init__(self, academy, action_space):
            super().__init__("random_agent", academy, action_space)

        def reset(self):
            pass

        def save(self, file):
            pass

        def restore(self, file) -> bool:
            pass

        def action(self, observation):
            return self.action_space.sample()

        def training_action(self, observation, progress):
            raise RuntimeError("Random agent can't be trained!")

        def teach_single(self, prev_observation, action, curr_observation, reward):
            pass

        def teach_survey(self, data):
            pass

    class TableMethodAgent(Agent):

        eps = 0.02

        def __init__(self, academy, observation_space, action_space):
            super().__init__("table_method_agent", academy, action_space)

            if type(action_space) is not Discrete or not (isinstance(observation_space, (Discrete, Box))):
                raise Exception(
                    "For table method agent only discrete action space and Box and Discrete observation space allowed "
                    "but was ", action_space, observation_space)

            if isinstance(observation_space, Box):

                # Environments with continuous observations handling
                self.obs_space_low = observation_space.low
                self.obs_space_high = observation_space.high
                self.n_states = 40

                state = []
                for s in observation_space.shape:
                    state += [self.n_states for _ in range(s)]
            else:
                state = [observation_space.n]

            state.append(action_space.n)
            self.Q = np.zeros(state)
            self.lr = .8
            self.gamma = .95

        def obs_to_state(self, obs) -> list:
            """ Maps an observation to state """
            env_low = self.obs_space_low
            env_high = self.obs_space_high
            # TODO: fix var overflow
            env_dx = (env_high - env_low) / self.n_states
            result = []
            for i in range(len(obs)):
                result.append(int((obs[i] - env_low[i]) / env_dx[i]))
            return result

        def reset(self):
            self.Q.fill(0)

        def save(self, file):
            utils.save_obj(self.Q, file)

        def restore(self, file) -> bool:
            q = utils.load_obj(file)
            if q is not None:
                self.Q = q
                return True
            return False

        def prepare_observation(self, observation):
            if isinstance(observation, (list, np.ndarray)):
                observation = self.obs_to_state(observation)
            else:
                # TODO: check for other observation types
                observation = [observation]

            return observation

        def action(self, observation):
            q_sa = self.get_q_values_for_observation(self.prepare_observation(observation))

            return np.argmax(q_sa)

        def get_q_values_for_observation(self, observation):
            q = self.Q
            for d in observation:
                q = q[d]

            if len(q.shape) > 1:
                raise RuntimeError("Observation does not match Q function!")

            return q

        def training_action(self, observation, progress: float):
            if np.random.uniform(0, 1) < (Academy.TableMethodAgent.eps * (1 - progress)):
                return np.random.choice(self.action_space.n)
            else:
                return self.action(observation)

        def teach_single(self, prev_observation, action, curr_observation, reward):
            q_prev = self.get_q_values_for_observation(self.prepare_observation(prev_observation))
            q_curr = self.get_q_values_for_observation(self.prepare_observation(curr_observation))

            update = (1 - self.lr) * q_prev[action] + self.lr * (reward + self.gamma * np.max(q_curr))
            loss = (q_prev[action] - update) ** 2
            q_prev[action] = update
            return False, loss

        def teach_survey(self, data):
            cumulative_loss = 0
            has_finish = False
            for item in data:
                finish, loss = self.teach_single(item[0], item[1], item[2], item[3])
                cumulative_loss += loss
                has_finish = True if has_finish or finish else False
            return has_finish, cumulative_loss

    class FeedForwardNetworkAgent(Agent):

        eps = 0.02

        def __init__(self, academy, observation_space, action_space):
            super().__init__("ffn_agent", academy, action_space)

            self.lr = .8
            self.gamma = .95

            input_shape = []
            if isinstance(observation_space, Box):
                input_shape = observation_space.high.shape
                self.obs_space_low = observation_space.low
                self.obs_space_high = observation_space.high

            # elif isinstance(observation_space, Discrete):
            #
            else:
                raise RuntimeError("Input space size wasn't defined for type %s" % type(observation_space))

            output_shape = action_space.n

            # with tf.name_scope("input"):
            i_shape = [1]
            i_shape.extend(input_shape)
            self.input = tf.placeholder(dtype=observation_space.dtype, shape=i_shape)

            l1_out_shape = 10 * output_shape
            w_shape = []
            w_shape.extend(input_shape)
            w_shape.append(l1_out_shape)

            w1 = tf.get_variable("w1", shape=w_shape, initializer=tf.random_uniform_initializer)
            b1 = tf.get_variable("b1", shape=[l1_out_shape], initializer=tf.random_uniform_initializer)

            w2 = tf.get_variable("w2", shape=[l1_out_shape, output_shape], initializer=tf.random_uniform_initializer)
            b2 = tf.get_variable("b2", shape=[output_shape], initializer=tf.random_uniform_initializer)

            l1 = tf.sigmoid(tf.nn.xw_plus_b(self.input, w1, b1))
            l_out = tf.sigmoid(tf.nn.xw_plus_b(l1, w2, b2))
            self.q_state = l_out
            self.q_sa_optimal = tf.argmax(self.q_state, 1)

            self.q_state_updated = tf.placeholder(shape=[1, output_shape], dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_state_updated, self.q_state))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)

            self.sess = tf.Session()
            self.saver = tf.train.Saver(var_list=[w1, b1, w2, b2])

            self.reset()

        def normalize_observation(self, obs):
            env_low = self.obs_space_low
            env_high = self.obs_space_high

            env_range = env_high - env_low

            result = []
            for i in range(len(obs)):
                result.append((obs[i] - env_low[i]) / env_range[i])

            return result

        def action(self, observation):
            observation = self.normalize_observation(observation)
            feed = {self.input: [observation]}

            action = self.sess.run(self.q_sa_optimal, feed_dict=feed)
            return action[0]

        def reset(self):
            self.sess.run(tf.global_variables_initializer())

        def training_action(self, observation, progress):
            if np.random.random() < (Academy.FeedForwardNetworkAgent.eps * (1 - progress)):
                action = np.random.choice(self.action_space.n)
                print("Random")
            else:
                action = self.action(observation)
            return action

        def teach_single(self, prev_observation, action, curr_observation, reward):

            curr_observation = self.normalize_observation(curr_observation)
            prev_observation = self.normalize_observation(prev_observation)
            # q_prev[action] = (1 - self.lr) * q_prev[action] + self.lr * (reward + self.gamma * np.max(q_curr))
            q_prev = self.sess.run(self.q_state, feed_dict={self.input: [prev_observation]})[0]
            q_curr = self.sess.run(self.q_state, feed_dict={self.input: [curr_observation]})[0]

            q_prev[action] = (1 - self.lr) * q_prev[action] + self.lr * (reward + self.gamma * np.max(q_curr))

            loss, _ = self.sess.run([self.loss, self.optimizer],
                                    feed_dict={self.input: [prev_observation], self.q_state_updated: [q_prev]})

            return False, loss

        def teach_survey(self, data):
            cumulative_loss = 0
            has_finish = False
            for item in data:
                finish, loss = self.teach_single(item[0], item[1], item[2], item[3])
                cumulative_loss += loss
                has_finish = True if has_finish or finish else False
            return has_finish, cumulative_loss

        def save(self, file):
            self.saver.save(self.sess, file)

        def restore(self, file) -> bool:
            if file is not None and os.path.exists(file + ".meta"):
                self.saver.restore(self.sess, file)
                return True
            else:
                return False


class Couch:

    def __init__(self):
        pass

    def train(self, environment: Environment, agent: Agent, episodes=1000, steps_per_episode=199, train_episode=False):
        agent.reset()
        # all_rewards = []
        progress_bar = trange(episodes)
        for i_episode in progress_bar:
            observation = environment.reset()
            episode_reward = 0.
            episode_loss = 0
            episode_data = []
            step = 0
            for step in range(steps_per_episode):
                action = agent.training_action(observation, i_episode / episodes)
                curr_observation, reward, done, info = environment.step(action)

                if train_episode:
                    episode_data.append((observation, action, curr_observation, reward))
                else:
                    result = agent.teach_single(observation, action, curr_observation, reward)
                    episode_loss += result[1]

                observation = curr_observation
                episode_reward += reward
                step += 1
                if done:
                    break
            if train_episode:
                result = agent.teach_survey(episode_data)
                episode_loss = result[1]

            progress_bar.set_description(
                "Reward per episode %s, cumul. loss %s" % (episode_reward, episode_loss / step))

    def validate(self, environment, agent, episodes=1000):
        all_rewards = []
        steps = []
        for i_episode in range(episodes):
            observation = environment.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done:
                action = agent.action(observation)
                observation, reward, done, info = environment.step(action)
                episode_reward += reward
                step += 1
            all_rewards.append(episode_reward)
            steps.append(step)

        success = [x for x in steps if environment.is_won(x, 0)]
        performance = len(success) / len(steps)

        plt.plot(all_rewards)
        plt.title("Cumulative reward per game. Won {}% of games".format(performance * 100))
        plt.show()

        return performance
