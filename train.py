import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

from q_function import QFunction
import agents


def train(env, method, q_function, max_episode, epsilon_, learning_rate_, discount_rate_, render=False, update_agent=True):

    total_rewards = []
    Agent = getattr(agents, method)

    for episode in range(max_episode):
        observations = env.reset()
        agent = Agent(q_function, observations, env.action_space.n, epsilon=epsilon_, learning_rate=learning_rate_, discount_rate=discount_rate_)
        done = False
        total_reward = 0
        if render: env.render()

        while not done:
            action = agent.decide_action(observations)
            observations, prev_reward, done, _ = env.step(action)
            prev_action = action
            total_reward += prev_reward
            if update_agent: agent.update_q_function(observations, prev_action, prev_reward)
            if render: env.render()

        total_rewards.append(total_reward)
        print(f"Episode {episode} finished, Total reward is {total_reward}")

    return total_rewards


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--round-digits", nargs="*", type=int, default=[3, 4])
    parser.add_argument("-m", "--methods", nargs="*", type=str, default=["QLearning", "SARSA"])
    parser.add_argument("-e", "--max-episode", type=int, default=10000)
    parser.add_argument("-p", "--epsilon", type=float, default=0.002)
    parser.add_argument("-l", "--learning-rate", type=float, default=0.3)
    parser.add_argument("-d", "--discount-rate", type=float, default=0.99)
    args = parser.parse_args()

    round_digits = args.round_digits
    max_episode = args.max_episode
    epsilon = args.epsilon
    learning_rate = args.learning_rate
    discount_rate = args.discount_rate

    results = []
    averaged_results = []

    for method in args.methods:

        video_dir = f"./videos/{method}"
        os.makedirs(video_dir, exist_ok=True)

        env = gym.make("MountainCar-v0")
        env = gym.wrappers.Monitor(env, video_dir, force=True)
        q_function = QFunction(round_digits)
        total_rewards = train(env, method, q_function, max_episode, epsilon, learning_rate, discount_rate)

        result = np.array(total_rewards)
        results.append(result)
        averaged_results.append(np.convolve(result, np.ones(100)/100, mode="valid"))

        env.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, xlabel="epoch", ylabel="total reward")
    for method, result, averaged_result, c in zip(args.methods, results, averaged_results, ["blue", "orange"]):
        ax.plot(result, linewidth=0.5, alpha=0.3, color=c,label=method)
        ax.plot(averaged_result, linewidth=1.5, alpha=1.0, color=c, label=f"{method} (weighted average)")
    figure_dir = f"./figures/"
    os.makedirs(figure_dir, exist_ok=True)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(figure_dir, "result.png"))


if __name__ == "__main__":
    main()