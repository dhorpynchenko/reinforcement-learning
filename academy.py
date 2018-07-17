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
    def teach_single(self, prev_observation, action, curr_observation, reward):
        pass

    @abstractmethod
    def teach_survey(self, data) -> bool:
        """
        Teach agent with data per episode
        :param data: list with episode data.
        data[0], data[2] - observation before and after action; data[1] - action, data[3] - reward
        :return True if algorithm converged, False otherwise
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

    def nn_agent(self, environment: Environment) -> Agent:
        return Academy.NNAgent(self, environment.observation_space(), environment.action_space())

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

            q_prev[action] = (1 - self.lr) * q_prev[action] + self.lr * (reward + self.gamma * np.max(q_curr))

        def teach_survey(self, data):
            for item in data:
                self.teach_single(item[0], item[1], item[2], item[3])

    class NNAgent(Agent):

        def __init__(self, academy, observation_space, action_space):
            super().__init__("nn_agent", academy, action_space)

            action_amount = action_space.n
            observation_size = observation_space.shape[0]

            # These lines establish the feed-forward part of the network used to choose actions
            input1 = tf.placeholder(dtype=observation_space.dtype, shape=(1, observation_size))
            W = tf.Variable(tf.random_uniform([observation_size, action_amount], 0, 0.01))
            Qout = tf.sigmoid(tf.matmul(input1, W))
            self.predict = tf.argmax(Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            nextQ = tf.placeholder(shape=[1, action_amount], dtype=tf.float32)
            loss = tf.reduce_sum(tf.square(nextQ - Qout))
            trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            self.optimizer = trainer.minimize(loss)

        def action(self, observation):
            return self.action_space.sample()

        def reset(self):
            pass

        def training_action(self, observation, progress):
            return self.action(observation)

        def teach_single(self, prev_observation, action, curr_observation, reward):
            pass

        def teach_survey(self, data):
            pass


class Couch:

    def __init__(self):
        pass

    def train(self, environment: Environment, agent: Agent, episodes=1000, steps_per_episode=201, train_episode=False):
        agent.reset()
        # all_rewards = []
        progress_bar = trange(episodes)
        for i_episode in progress_bar:
            observation = environment.reset()
            episode_reward = 0.
            episode_data = []
            for step in range(steps_per_episode):
                action = agent.training_action(observation, i_episode / episodes)
                curr_observation, reward, done, info = environment.step(action)

                if train_episode:
                    episode_data.append((observation, action, curr_observation, reward))
                else:
                    agent.teach_single(observation, action, curr_observation, reward)

                observation = curr_observation
                episode_reward += reward
                if done:
                    break
            progress_bar.set_description("Episode %s, reward per episode %s" % (i_episode, episode_reward))
            if train_episode:
                agent.teach_survey(episode_data)

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
        plt.title("Cumulative reward per game. Won {}% of games".format(performance* 100))
        plt.show()

        return performance
