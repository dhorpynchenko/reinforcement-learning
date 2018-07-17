import gym


class Environment:

    @staticmethod
    def frozen_lake_env(render_fraction):
        return Environment("FrozenLake-v0", render_fraction)

    @staticmethod
    def mountain_car_env(render_fraction):
        return Environment("MountainCar-v0", render_fraction)

    @staticmethod
    def cart_pole(render_fraction):
        return Environment("CartPole-v1", render_fraction)

    def __init__(self, name, render_fraction):
        """
        :param name: name of gym environment
        :param render_fraction: percent of frames to render 0..1
        """
        # Init vars
        self._render_skip_frames = 1
        self._render_frames_skipped = 0
        self.set_render_fraction(render_fraction)

        self.name = name
        self.env = gym.make(name)
        self.env.seed(0)
        gym.logger.set_level(gym.logger.INFO)

        print("Creating environment %s" % name)
        print("Action space ", self.env.action_space)
        print("Observable space ", self.env.observation_space)
        print("Reward bounds ", self.env.reward_range)

    def set_render_fraction(self, render_fraction):
        if render_fraction <= 0:
            self._render_skip_frames = -1
        else:
            if render_fraction > 1:
                render_fraction = 1
            self._render_skip_frames = 1 // render_fraction - 1

        self._render_frames_skipped = 0

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

    def reward_range(self):
        return self.env.reward_range

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def is_won(self, total_steps, last_reward):
        return total_steps < self.env._max_episode_steps
