import utils
from environment import Environment
from academy import Academy, Couch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Environment.mountain_car_env(0)
    academy = Academy()
    agent = academy.table_method_agent(env)
    # restored = academy.restore_agent_settings(agent, env)
    # if restored:
    #     print("Restored from save point")

    couch = Couch()

    steps = couch.train(env, agent, episodes=150000, train_episode=False)
    academy.save_agent_settings(agent, env)
    utils.save_obj(steps, "AcademySave/log/steps_approx")

    env.set_render_fraction(0.5)
    performance, steps = couch.validate(env, agent, episodes=100)

    plt.plot(steps)
    plt.title("Cumulative reward per game. Won {}% of games".format(performance * 100))
    plt.show()

    env.close()
