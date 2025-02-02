import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from agent import LunarLanderAgent

model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("src", "models", "default")
model_name = sys.argv[2] if len(sys.argv) > 2 else "lunar_lander_model"
agent = LunarLanderAgent(
    model_path=model_path,
    model_name=model_name,
)
DENOM = 10

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

lengths = [float(l) for l in agent.episode_lengths]
x = np.arange(len(lengths)//DENOM)*DENOM
y = [sum(lengths[i*DENOM:i*DENOM+DENOM])/DENOM for i in range(len(lengths)//DENOM)]
coeffs = np.polyfit(x, y, deg=1)
poly_eq = np.poly1d(coeffs)
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = poly_eq(x_smooth)

axs[0].scatter(x, y, color="red")
axs[0].plot(x_smooth, y_smooth)
axs[0].set_title(f"Averages of every {DENOM} episode lengths")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Length")

x = np.arange(len(agent.rewards)//DENOM)*DENOM
y = [float(sum(agent.rewards[i*DENOM:i*DENOM+DENOM])/DENOM) for i in range(len(agent.rewards)//DENOM)]
coeffs = np.polyfit(x, y, deg=1)
poly_eq = np.poly1d(coeffs)
x_smooth = np.linspace(x.min(), x.max(), 200)
y_smooth = poly_eq(x_smooth)

axs[1].scatter(x, y, color="red")
axs[1].plot(x_smooth, y_smooth)
axs[1].set_title(f"Averages of every {DENOM} rewards")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Reward")

plt.tight_layout()
plt.show()