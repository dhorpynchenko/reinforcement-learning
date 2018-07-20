from environment import Environment
from academy import Academy, Couch

if __name__ == '__main__':
    env = Environment.mountain_car_env(0)
    academy = Academy()
    agent = academy.feed_forward_network_agent(env)
    # restored = academy.restore_agent_settings(agent, env)
    # if restored:
    #     print("Restored from save point")

    couch = Couch()

    couch.train(env, agent, episodes=1000, train_episode=False)
    academy.save_agent_settings(agent, env)

    env.set_render_fraction(0.5)
    couch.validate(env, agent, episodes=100)

    env.close()
