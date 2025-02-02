from collections import defaultdict, deque
import json
import os
import random
import time
import gymnasium as gym
from keras import Sequential, Input
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np


class LunarLanderAgent:
    def __init__(
        self,
        env: gym.Env = gym.make("LunarLander-v2"),
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        n_episodes: int = 100,
        model_path: str = os.path.join("src", "models", "default"),
        model_name: str = "lunar_lander_model",
        load: bool = True,
        weights_path: str = "",
    ):
        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes

        self.memory = []
        self.rewards = []
        self.length_queue = []
        self.model_path = model_path
        self.model_name = model_name

        self.model = Sequential([
            Input(shape=(8,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(4, activation='linear'),
        ])
        self.optimizer = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimizer, loss='mse')

        if load:
            self.load_model(weights_path)

    def get_action(self, state: tuple[int, int, bool]) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        else:
            action_values = self.model(np.reshape(state, [1, 8]))
            return np.argmax(action_values)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory[-1000:], batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        file_path = os.path.join(self.model_path, self.model_name)
        os.makedirs(os.path.join(self.model_path, "weights"), exist_ok = True)
        
        self.model.save(file_path + ".keras")
        self.model.save_weights(os.path.join(self.model_path, "weights", f"{self.model_name}_{time.strftime('%Y%m%d-%H%M%S')}.weights.h5"))
        # with open(self.model_path + ".json", "w") as f:
        #     json.dump({
        #         # "rewards": self.rewards,
        #         # "lengths": [np.float32(l) for l in self.length_queue],
        #         "memory": [[m.tolist() if type(m) == np.ndarray else m for m in e] for e in self.memory],
        #     }, f, indent=4)
        np.save(file_path + "_rewards.npy", np.array(self.rewards))
        np.save(file_path + "_lengths.npy", np.array(self.length_queue))
        # np.save(self.model_path + "_memory.npy", np.array(self.memory))
        print("Model saved successfully!")

    def load_model(self, weights_path):
        file_path = os.path.join(self.model_path, self.model_name)
        if weights_path:
            self.model.load_weights(weights_path)
            print("Weights loaded!")
        else:
            if os.path.exists(file_path + ".keras"):
                self.model = load_model(file_path + ".keras")
                print("Model loaded!")
            else:
                print("Model file does not exist")

        # if os.path.exists(self.model_path + ".json"):
        #     try:
        #         with open(self.model_path + ".json", "r") as f:
        #             hyperparameters = json.load(f)
        #             # self.rewards = hyperparameters["rewards"]
        #             # self.length_queue = hyperparameters["lengths"]
        #             self.memory = [tuple(np.array(m, dtype=np.float32) if type(m) == list else m for m in e) for e in hyperparameters["memory"]]
        #     except:
        #         print("json exists but hyperparameters could not be loaded")


        if os.path.exists(file_path + "_rewards.npy"):
            self.rewards = list(np.load(file_path + "_rewards.npy"))
            print(f"Loaded reward history of {len(self.rewards)} episodes!")
        else:
            print("Reward history file does not exist")

        if os.path.exists(file_path + "_lengths.npy"):
            self.length_queue = list(np.load(file_path + "_lengths.npy"))
            print(f"Loaded length history of {len(self.length_queue)} episodes!")
        else:
            print("Length history file does not exist")
        
        # if os.path.exists(self.model_path + "_memory.npy"):
        #     self.memory = list(np.load(self.model_path + "_memory.npy"))
        #     print(f"Loaded memory of {len(self.memory)} episodes.")
        # else:
        #     print("memory file does not exist")