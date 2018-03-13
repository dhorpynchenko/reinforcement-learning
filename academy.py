from abc import abstractmethod, ABCMeta

from gym.spaces import Discrete

from environment import Environment
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Agent(metaclass=ABCMeta):

    def __init__(self, name, academy, action_space):
        self.action_space = action_space
        self.name = name
        self.academy = academy

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
    def teach(self, prev_observation, action, curr_observation, reward):
        pass


class Academy:

    def __init__(self, save_folder="./AcademySave"):
        self.save_folder = save_folder

    def random_agent(self, environment: Environment) -> Agent:
        return Academy.RandomAgent(self, environment.action_space())

    def table_method_agent(self, environment: Environment) -> Agent:
        observation_space = environment.observation_space()
        action_space = environment.action_space()

        if type(observation_space) is not Discrete or type(action_space) is not Discrete:
            raise Exception("For table method agent only discrete observation and action space allowed but was ",
                            observation_space, action_space)

        return Academy.TableMethodAgent(self, observation_space, action_space)

    def nn_agent(self, environment: Environment) -> Agent:
        return Academy.NNAgent(self, environment.observation_space(), environment.action_space())

    def function_agent(self, environment: Environment) -> Agent:
        pass

    def save_agent_settings(self, agent: Agent, environment: Environment):
        pass

    class RandomAgent(Agent):
        """
        Acts randomly
        """

        def reset(self):
            pass

        def training_action(self, observation, progress):
            pass

        def teach(self, prev_observation, action, curr_observation, reward):
            pass

        def __init__(self, academy, action_space):
            super().__init__("random_agent", academy, action_space)

        def action(self, observation):
            return self.action_space.sample()

    class TableMethodAgent(Agent):

        def reset(self):
            self.Q.fill(0)

        def __init__(self, academy, observation_space, action_space):
            super().__init__("table_method_agent", academy, action_space)
            self.Q = np.zeros([observation_space.n, action_space.n])
            self.lr = .8
            self.y = .95

        def action(self, observation):
            return np.argmax(self.Q[observation, :])

        def training_action(self, observation, progress: float):
            return np.argmax(
                self.Q[observation, :] + np.random.randn(1, self.action_space.n) * (1. / (progress * 100 + 1)))

        def teach(self, prev_observation, action, curr_observation, reward):
            self.Q[prev_observation, action] = self.Q[prev_observation, action] + self.lr * (
                    reward + self.y * np.max(self.Q[curr_observation, :]) - self.Q[prev_observation, action])

    class FunctionAgent(Agent):

        def reset(self):
            pass

        def training_action(self, observation, progress):
            pass

        def teach(self, prev_observation, action, curr_observation, reward):
            pass

        def __init__(self, academy, action_space):
            super().__init__("function_agent", academy, action_space)

        def action(self, observation):
            pass

    class NNAgent(Agent):

        def reset(self):
            pass

        def training_action(self, observation, progress):
            return self.action(observation)

        def teach(self, prev_observation, action, curr_observation, reward):
            pass

        def __init__(self, academy, observation_space, action_space):
            super().__init__("nn_agent", academy, action_space)
            # These lines establish the feed-forward part of the network used to choose actions
            input1 = tf.placeholder(dtype=observation_space.dtype, shape=(1, observation_space.shape[0]))
            W = tf.Variable(tf.random_uniform([observation_space.shape[0], action_space.n], 0, 0.01))
            print(input1, W)
            Qout = tf.matmul(input1, W)
            predict = tf.argmax(Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            nextQ = tf.placeholder(shape=[1, action_space.n], dtype=tf.float32)
            loss = tf.reduce_sum(tf.square(nextQ - Qout))
            trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            updateModel = trainer.minimize(loss)

        def action(self, observation):
            return self.action_space.sample()


class Couch:

    def __init__(self):
        pass

    def train(self, environment: Environment, agent: Agent, episodes=1000, steps_per_episode=100):
        agent.reset()
        all_rewards = []
        for i_episode in range(episodes):
            print("Episode {}".format(i_episode))
            observation = environment.reset()
            episode_reward = 0.
            for step in range(steps_per_episode):
                action = agent.training_action(observation, i_episode / episodes)
                curr_observation, reward, done, info = environment.step(action)
                agent.teach(observation, action, curr_observation, reward)
                observation = curr_observation
                episode_reward += reward
                if done:
                    break
            print("Episode reward ", episode_reward)
            all_rewards.append(episode_reward)
        # plt.plot(all_rewards)
        # plt.show()

    def validate(self, environment, agent, episodes=1000):
        all_rewards = []
        for i_episode in range(episodes):
            observation = environment.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.action(observation)
                observation, reward, done, info = environment.step(action)
                episode_reward += reward
            all_rewards.append(episode_reward)
        plt.plot(all_rewards)
        plt.title("Cumulative reward per game. Won {}% of games".format(
            len([filter(lambda r: r > 0, all_rewards)]) / len(all_rewards) * 100))
        plt.show()

        return 'Bellissimo!'
