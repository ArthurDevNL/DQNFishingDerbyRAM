import gym
import numpy as np
import pandas as pd
env = gym.make('FishingDerby-ram-v4')
env.seed(42)


def phi(x):

	line_x = int(x[33])
	line_y = int(x[68])

	fish6_x = int(x[69])
	fish6_y = 252

	v1 = line_x - fish6_x
	v2 = fish6_y - line_y

	v5 = 0 if x[114] == 0 else 1
	return np.array([v1, v2, v5])

observation = env.reset()
state_size = phi(observation).shape[0]

def get_action(obs, obs_):
	line_x = int(obs[33])
	line_y = int(obs[68])

	line_x_ = int(obs_[33])
	line_y_ = int(obs_[68])

	dx = line_x - line_x_
	dy = line_y - line_y_

	if dx == 0 and dy == 0:
		return 0

	a = []

	if dy < 0:
		a.append(5)
	elif dy > 0:
		a.append(2)
	elif dx < 0:
		a.append(4)
	elif dx > 0:
		a.append(3)

	return a[np.random.randint(0,len(a))]

pending_reward_idx = 115
last_reward_frames = 0
caught_fish_idx = 113
def get_reward(obs, obs_):

	global last_reward_frames
	if last_reward_frames > 0:
		last_reward_frames -= 1
		return 0

	# Only give reward if fish with value 4 is caught
	pending_reward = obs_[pending_reward_idx]
	if pending_reward > 0:
		last_reward_frames = pending_reward + 1
		return pending_reward + 1

	return 0

D = []
episode = 0
while episode < 100:
	print('Episode', episode)
	observation = env.reset()

	done = False
	while not done:
		# env.render()
		s1, s2, s3 = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = 0

		# Take the chosen action
		observation_, reward, done, info = env.step(action)

		# Store the tuple
		s1_, s2_, s3_ = phi(observation_)

		action_taken = get_action(observation, observation_)
		reward = get_reward(observation, observation_)

		if reward == 0:
			reward = -0.01

		D.append((episode, s1, s2, s3, action_taken, reward, s1_, s2_, s3_))

		observation = observation_

	episode += 1

pd.DataFrame(D, columns=['episode', 's1', 's2', 's3', 'action', 'reward', 's1_', 's2_', 's3_']).to_csv('data.csv')