import os
from matplotlib import pyplot as plt
import numpy as np
from agent import LunarLanderAgent

models = []
paths_info = [(os.path.join("src", "models", d), f[:-6]) for d in os.listdir(os.path.join("src", "models")) for f in os.listdir(os.path.join("src", "models", d)) if os.path.isfile(os.path.join("src", "models", d, f)) and f[-6:] == ".keras"]
for path, name in paths_info:
    models.append(LunarLanderAgent(
        model_path=path,
        model_name=name
    ))
DENOM = 50

plt.figure(figsize=(10,6))

for i, model in enumerate(models):
    x = np.arange(len(model.rewards)//DENOM)*DENOM
    y = [float(sum(model.rewards[i*DENOM:i*DENOM+DENOM])/DENOM) for i in range(len(model.rewards)//DENOM)]
    plt.plot(x, y, label=f"α={model.lr} γ={model.discount_factor} ε decay={model.epsilon_decay}")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"Reward Obtained During Training By {len(models)} Models")
plt.legend()
plt.grid(True)
plt.show()