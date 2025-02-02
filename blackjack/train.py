import gymnasium as gym
import numpy as np
from tqdm import tqdm
from agent import BlackjackAgent

# hyperparameters
learning_rate = 0.01
n_episodes = 100_000

agent = BlackjackAgent(
    n_episodes=n_episodes,
    learning_rate=learning_rate,
    load=True,
)

for episode, i in enumerate(tqdm(range(n_episodes), leave=True)):
    obs, info = agent.env.reset()
    done = False
    total_reward = 0

    # play one episode
    while not done:
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = agent.env.step(action)

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

    agent.rewards.append(total_reward)
    agent.length_queue.append(agent.env.length_queue[-1])

    # save occasionally
    if ((i+1)%1000 == 0):
        agent.save_model()

agent.save_model()
agent.env.close()