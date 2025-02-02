from collections import defaultdict
import gymnasium as gym
from keras import Sequential, Input
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf
import time


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env = gym.make("Blackjack-v1", sab=False),
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        n_episodes: int = 100,
        model_path: str = os.path.join("blackjack", "blackjack_model"),
        load: bool = True,
    ):
        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.n_episodes = n_episodes

        self.rewards = []
        self.length_queue = []
        self.model_path = model_path

        if load:
            self.load_model()
        else:
            self.model = Sequential([
                Input(shape=(3,)),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(2, activation='linear'),
            ])
            self.model.compile(optimizer='adam', loss='mse')

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        q_values = self.model(np.array(obs)[np.newaxis])
        return int(np.argmax(q_values))

    def save_model(self):
        self.model.save(self.model_path + ".keras")
        self.model.save_weights(os.path.join("blackjack", "weights", f"blackjack_model_{time.strftime("%Y%m%d-%H%M%S")}.weights.h5"))
        np.save(self.model_path + "_rewards.npy", np.array(self.rewards))
        np.save(self.model_path + "_lengths.npy", np.array(self.length_queue))
        print("Model saved successfully!")

    def load_model(self):
        if os.path.exists(self.model_path + ".keras"):
            self.model = load_model(self.model_path + ".keras")
            print("Model loaded!")
        else:
            print("model file does not exist")

        if os.path.exists(self.model_path + "_rewards.npy"):
            self.rewards = list(np.load(self.model_path + "_rewards.npy"))
            print(f"Loaded reward history of {len(self.rewards)} episodes.")
        else:
            print("reward history file does not exist")

        if os.path.exists(self.model_path + "_lengths.npy"):
            self.length_queue = list(np.load(self.model_path + "_lengths.npy"))
            print(f"Loaded length history of {len(self.length_queue)} episodes.")
        else:
            print("length history file does not exist")