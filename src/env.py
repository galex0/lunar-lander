# installing gym properly, stuck on pip install gymnasium[box2d] | conda install swig 
# as conda hasn't downloaded yet.
# https://www.youtube.com/watch?v=gMgj4pSHLww

# installing conda, already downloaded
# might have to enter windows and change partitions with gnome, anaconda keeps failing to download
# https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

import gym

env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
)