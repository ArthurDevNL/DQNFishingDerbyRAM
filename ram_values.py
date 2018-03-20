import gym
from time import sleep
import os
import numpy as np
from collections import Counter

env = gym.make('FishingDerby-ram-v4')
env.seed(42)

fishes = [68]

fish_min = 255
fish_max = 0

# Fishies swim between 18 and 133
# Shark swims between 19 and 105

# Opp rod x between 4 and 15
# Opp line x between 66 and 139
# Opp line y between 200 and 252

# Rod x between 4 and 15
# Line x between 19 and 98
# Line y between 200 and 252

observation = env.reset()

i = 0
while True:
	# env.render()

	action = env.action_space.sample()
	# action = 3
	observation, reward, done, info = env.step(action)
	if done:
		print(fish_min, fish_max)
		observation = env.reset()

	fish_xs = observation[fishes]
	if max(fish_xs) > fish_max:
		fish_max = max(fish_xs)

	if min(fish_xs) < fish_min:
		fish_min = min(fish_xs)

env.close()

shark_x_idx = 75

fish2_top_x_idx = 74
fish2_bot_x_idx = 73

fish4_top_x_idx = 72
fish4_bot_x_idx = 71

fish6_top_x_idx = 70
fish6_bot_x_idx = 69

opp_rod_x = 22
opp_line_x = 33
opp_line_y = 68

rod_x = 21
line_x = 32
line_y = 67
