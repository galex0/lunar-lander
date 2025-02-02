import os
import sys
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from agent import LunarLanderAgent

# hyperparameters
learning_rate = 0.001
discount_factor = 0.99
n_episodes = 3000
batch_size = 32

model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("src", "models", "default")
model_name = sys.argv[2] if len(sys.argv) > 2 else "lunar_lander_model"

agent = LunarLanderAgent(
    n_episodes=n_episodes,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    load=True,
    model_path=model_path,
    model_name=model_name,
    # weights_path="src/models/default/weights/lunar_lander_model_20250201-052703.weights.h5"
)
state_size = agent.env.observation_space.shape[0]


for episode, i in enumerate(tqdm(range(n_episodes), leave=True)):
    state, info = agent.env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    # play one episode
    while not done:
        action = agent.get_action(state)

        next_state, reward, terminated, truncated, info = agent.env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        done = terminated or truncated

        state = next_state
        total_reward += reward

        if done:
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    agent.rewards.append(total_reward)
    agent.length_queue.append(agent.env.length_queue[-1])

    # save occasionally
    if ((i+1)%10 == 0):
        agent.save_model()

agent.env.close()