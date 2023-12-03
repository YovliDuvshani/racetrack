from agent import Agent
from config import RELATIVE_TRACK_PATH
from env import Env

from policy_manager import PolicyManager
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = Env(RELATIVE_TRACK_PATH)
    policy_manager = PolicyManager(env)

    agent = Agent(env, policy_manager)

    agent.monte_carlo_control()
    v = agent.v()

    plt.imshow(v[:, :, 5, 6])

    optimal_track = [t_s.state.position for t_s in env.generate_episode(agent.policy)]

    fig, ax = plt.subplots()
    ax.plot(plt.imshow(env.map))
    ax.plot(plt.scatter([pos.y for pos in optimal_track], [pos.x for pos in optimal_track]))
    plt.show()
