import gym
from time import sleep
import os
import numpy as np
from collections import Counter
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

env = gym.make('FishingDerby-ram-v4')
env.seed(42)

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

counter = Counter()

exclude = set([69,70,71,72,73,74,75, 14, 22, 33, 37, 66, 68, 82, 120, 13, 21, 121])

def clear():
	os.system('clear')

print(env.unwrapped.get_action_meanings())

def print_ram(ram):
	ram = ram.reshape((8,16))
	for i in range(8):
		s = ''
		for j in range(16):
			s += str(ram[i,j]) + '\t'
		print(s)

observation = env.reset()
observation_ = observation


#[13, 21, 32, 36, 60, 65, 67, 121]

pending_reward_opp_idx = 115
caught_fish_opp_idx = 114

# Contains the number of the fish that was caught (6 to 1 from top to bottom)
caught_fish_idx = 112
pending_reward_idx = 113

last_b = False
had_fish = False

i = 0
while True:
	env.render()

	# sleep(0.5)

	action = env.action_space.sample()
	# action = 3
	observation, reward, done, info = env.step(action)
	clear()

	r = np.subtract(observation, observation_)
	print_ram(r)
	print(observation[113]) #114
	print_ram(observation_)

	# plt.imshow(observation)
	# plt.grid(True)
	# plt.savefig('test')

	# if observation_[caught_fish_idx] > 0 and observation[caught_fish_idx] == 0 and observation[pending_reward_idx] == 0:
	# 	print("RAAwwrrrr miam")
	# 	input()

	input()
	# if observation[caught_fish_idx]

	# if observation[caught_fish_idx] > 0 or last_b:
	# 	input()
	# 	last_b = True

	observation_ = observation

env.close()