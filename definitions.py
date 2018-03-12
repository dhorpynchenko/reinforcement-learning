import gym
import tensorflow as tf
import json


class Environment:

    def __init__(self, name, render_fraction=0.25):
        """
        :param name: name of gym environment
        :param render_fraction: percent of frames to render 0..1
        """
        if render_fraction <= 0:
            self._render_skip_frames = -1
        else:
            if render_fraction > 1:
                render_fraction = 1
            self._render_skip_frames = 1 // render_fraction - 1

        self._render_frames_skipped = 0
        self.name = name
        self.env = gym.make(name)
        self.env.seed(0)
        gym.logger.set_level(gym.logger.INFO)

        print("Action space ", self.env.action_space)
        print("Observable space ", self.env.observation_space)

    def step(self, action):
        result = self.env.step(action)
        if self._render_skip_frames >= 0:
            if self._render_frames_skipped >= self._render_skip_frames:
                self.env.render()
                self._render_frames_skipped = 0
            else:
                self._render_frames_skipped += 1

        return result

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()


class Agent:

    def __init__(self, name, action_space):
        self.action_space = action_space
        self.name = name

    def action(self, observation):
        pass

    def reset(self):
        pass


class Academy:

    def __init__(self, environment, save_folder="./AcademySave"):
        self.env = environment
        self.save_folder = save_folder

    def random_agent(self):
        return Academy.RandomAgent(self.env.action_space())

    def nn_agent(self):
        pass

    def function_agent(self):
        pass

    def save_agent_settings(self, agent, environment):
        pass

    class RandomAgent(Agent):
        """
        Acts randomly
        """

        def __init__(self, action_space):
            super().__init__("random_agent", action_space)

        def action(self, observation):
            return self.action_space.sample()

    class FunctionAgent(Agent):

        def __init__(self, action_space):
            super().__init__("function_agent", action_space)

        def action(self, observation):
            pass

    class NNAgent(Agent):

        def __init__(self, action_space):
            super().__init__("nn_agent", action_space)

        def action(self, observation):
            pass


class Couch:

    def __init__(self):
        pass

    def train(self, environment, agent):
        agent.reset()
        for i_episode in range(20):
            print("Episode {}".format(i_episode))
            observation = environment.reset()
            t = 0
            while 1:
                action = agent.action(observation)
                observation, reward, done, info = environment.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
                t = t + 1

    def validate(self, environment, agent):
        return 'Bellissimo!'
