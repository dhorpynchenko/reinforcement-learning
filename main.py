import argparse

import utils
from environment import Environment
from academy import Academy, Couch
import matplotlib.pyplot as plt

DEFAULT_EPISODES_MLP = 1000
DEFAULT_EPISODES_TABLE = 20000

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Q-learning agent on "Mountain Car" domain')

    parser.add_argument('--agent', required=False, choices=['table', 'mlp'], default='mlp', help='Agent to train')

    parser.add_argument('--episodes', required=False,
                        default=-1,
                        type=int,
                        help='Amount of episodes to run. Default is %s for table agent and %s for MLP agent' % (
                            DEFAULT_EPISODES_TABLE, DEFAULT_EPISODES_MLP))

    parser.add_argument('--r_train', required=False,
                        default=0.3,
                        type=float,
                        help='Fraction of frames to render during train [0, 1]. 0 for no rendering. Default is 0.3 (30%)')

    parser.add_argument('--r_valid', required=False,
                        default=0.5,
                        type=float,
                        help='Fraction of frames to render during validation [0, 1]. 0 for no rendering. Default is 0.5 (50%)')

    args = parser.parse_args()
    print(parser.description)

    env = Environment.mountain_car_env(args.r_train)
    print("Rendering {}% of frames".format(args.r_train * 100))
    academy = Academy()
    if args.agent == 'table':
        agent = academy.table_method_agent(env)
    else:
        agent = academy.feed_forward_network_agent(env)

    print("Agent %s" % agent.name)
    # restored = academy.restore_agent_settings(agent, env)
    # if restored:
    #     print("Restored from save point")

    ep = 0
    if args.episodes and args.episodes > 0:
        ep = args.episodes
    else:
        ep = DEFAULT_EPISODES_TABLE if args.agent == 'table' else DEFAULT_EPISODES_MLP
    print("Episodes count %s" % ep)

    couch = Couch()

    steps = couch.train(env, agent, episodes=ep, train_episode=False)
    academy.save_agent_settings(agent, env)
    utils.save_obj(steps, "AcademySave/log/steps_approx")

    env.set_render_fraction(args.r_valid)
    print("Validation. Rendering {}% of frames".format(args.r_valid * 100))
    performance, steps = couch.validate(env, agent, episodes=100)

    plt.plot(steps)
    plt.title("Cumulative reward per game. Won {}% of games".format(performance * 100))
    plt.show()

    env.close()
