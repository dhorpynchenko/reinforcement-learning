from environment import Environment
from academy import Academy, Couch

if __name__ == '__main__':
    env = Environment.frozen_lake_env(0)
    academy = Academy()
    agent = academy.table_method_agent(env)
    couch = Couch()

    couch.train(env, agent, episodes=50000, train_episode=True)
    couch.validate(env, agent, episodes=100)
    # academy.save_agent_settings(agent, env)

    env.close()
