import os
import random
import time
import pickle
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
        weights_path: str = "",
        load: bool = True,
        training: bool = True,
    ):
        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(env, n_episodes)

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        self.training = training

        self.memory = []
        self.rewards = []
        self.episode_lengths = []
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
        """
        With probability 1-epsilon return index of highest model-approximated q-value in the state.
        With probability epsilon choose an action randomly.
        """
        if self.training and np.random.rand() <= self.epsilon:
            return random.randrange(4)
        else:
            action_values = self.model(np.reshape(state, [1, 8]))
            return np.argmax(action_values)
        
    def remember(self, state, action, reward, next_state, done):
        """
        Memorize data for later training use.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Train model on memorized data using batch_size examples from last 1000 examples. 
        """
        minibatch = random.sample(self.memory[-1000:], batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.discount_factor * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
        self.model.fit(np.vstack(states), np.vstack(targets), batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """
        Update the .keras model file.
        Update the .pkl hyperparameter file.
        Add new .weights.h5 file to the weights dir (ie. create a checkpoint).
        """
        file_path = os.path.join(self.model_path, self.model_name)
        os.makedirs(os.path.join(self.model_path, "weights"), exist_ok = True)
        self.model.save(file_path + ".keras")
        self.model.save_weights(os.path.join(self.model_path, "weights", f"{self.model_name}_{time.strftime('%Y%m%d-%H%M%S')}.weights.h5"))
        hyperparameters = {
            "learning_rate": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "memory": self.memory,
            "rewards": self.rewards,
            "episode_lengths": self.episode_lengths,
        }
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(hyperparameters, f)
        print("Model saved successfully!")

    def load_model(self, weights_path):
        """
        Load model from the .keras model file.
        Load hyperparameters from the .pkl hyperparameter file.
        Load weights instead of model if weights_path is provided.
        """
        file_path = os.path.join(self.model_path, self.model_name)
        if weights_path:
            self.model.load_weights(weights_path)
            print("Weights loaded!")
        else:
            if os.path.exists(file_path + ".keras"):
                self.model = load_model(file_path + ".keras")
                print("Model loaded!")
            else:
                print("Model file not found")
        
        if os.path.exists(file_path + ".pkl"):
            with open(file_path + ".pkl", "rb") as f:
                hyperparameters = pickle.load(f)
            self.lr = hyperparameters["learning_rate"]
            self.discount_factor = hyperparameters["discount_factor"]
            self.epsilon = hyperparameters["epsilon"]
            self.epsilon_decay = hyperparameters["epsilon_decay"]
            self.epsilon_min = hyperparameters["epsilon_min"]
            self.memory = hyperparameters["memory"]
            self.rewards = hyperparameters["rewards"]
            self.episode_lengths = hyperparameters["episode_lengths"]
            print("Hyperparameters loaded!")
        else:
            print("Hyperparameters file not found")