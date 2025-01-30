from collections import defaultdict
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from agent import BlackjackAgent

# hyperparameters
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes), leave=True):
    obs, info = env.reset()
    done = False
    total_reward = 0

    # play one episode
    while not done:
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        target = np.empty((1,2))
        if done:
            target[0][action] = reward
        else:
            next_q_values = agent.model(np.array(next_obs)[np.newaxis])
            target[0][action] = reward + agent.discount_factor * np.max(next_q_values)

        agent.model.fit(np.array(obs)[np.newaxis], target, verbose=0)

        obs = next_obs
        total_reward += reward

    agent.decay_epsilon()
    agent.rewards.append(total_reward)

# visualize the episode rewards, episode length and training error in one figure
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# np.convolve will compute the rolling mean for 100 episodes

# print(np.array(env.return_queue).flatten(), "\n")
# print(len(env.length_queue), "\n")
# print(len(agent.training_error))
# axs[0].plot(np.convolve(np.array(env.return_queue).flatten(), np.ones(100)))
# axs[0].set_title("Episode Rewards")
# axs[0].set_xlabel("Episode")
# axs[0].set_ylabel("Reward")

x = np.arange(len(env.length_queue))
coeffs = np.polyfit(x, np.array(env.length_queue).flatten(), deg=1)
poly_eq = np.poly1d(coeffs)
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = poly_eq(x_smooth)

axs[1].scatter(x, env.length_queue, color="red")
axs[1].plot(x_smooth, y_smooth)
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

x = np.arange(len(agent.rewards))
coeffs = np.polyfit(x, agent.rewards, deg=1)
poly_eq = np.poly1d(coeffs)
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = poly_eq(x_smooth)

axs[2].scatter(x, agent.rewards, color="red")
axs[2].plot(x_smooth, y_smooth)
axs[2].set_title("Cubic Fit: Rewards")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Reward")

print(agent.rewards[-1])


plt.tight_layout()
plt.show()

env.close()