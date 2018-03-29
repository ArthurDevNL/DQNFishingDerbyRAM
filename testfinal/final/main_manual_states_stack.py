import gym
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import *
from keras.utils import to_categorical
from collections import deque
from itertools import islice
import random
import numpy as np
from time import sleep
import pandas as pd
env = gym.make('FishingDerby-ram-v4')
env.seed(42)

# mn = min, mx = max
def rescale(v, mn, mx):
	v = min(v - mn, mx - mn)
	return float(v / (mx - mn))

def phi(x):

	features = []

	min_x = 18
	max_x = 100

	# Fishies swim between 18 and 133
	# fishes = [69, 70, 71, 72, 73, 74]
	# for f in fishes:
	# 	v = rescale(x[f], min_x, max_x)
	# 	features.append(v)

	# Just add the fish of value 6
	v = rescale(x[74], min_x, max_x)
	features.append(v)

	# Shark swims between 19 and 105
	shark_x = 75
	v = rescale(x[shark_x], min_x, max_x)
	features.append(v)

	# Line x between 19 and 98
	line_x = 32
	v = rescale(x[line_x], min_x, max_x)
	features.append(v)

	# Line y between 200 and 252
	line_y = 67
	v = min(max(x[line_y] - 200, 0), 54)
	one_hot = to_categorical(v, num_classes=55)
	features.extend(one_hot)

	caught_fish_idx = 112
	v = 0 if x[caught_fish_idx] == 0 else 1
	# v = rescale(x[caught_fish_idx], 0,6)
	features.append(v)

	return np.array(features)

observation = env.reset()
state_size = phi(observation).shape[0]

actions = [0,1,2,3,4,5]#,3,4,5]
n_actions = 6 #env.action_space.n

print(env.unwrapped.get_action_meanings())
print('State size:', state_size)

test = True
load_model = True

hist_size = 3

# Initialize value function
model = Sequential()
model.add(Flatten(input_shape=(state_size, hist_size)))
model.add(Dense(64, input_dim=state_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(n_actions))

if load_model:
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")

opt = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=opt)

# Initialize dataset D
D = deque(maxlen=100000)

e = 1.0 if not test else 0
e_decay_frames = 300000
e_min = 0.1

gamma = 0.99

update_freq = 4
counter = 0

replay_mem_size = 50000
batch_size = 32

pending_reward_idx = 114

last_reward_frames = 0
def get_reward(obs):
	global last_reward_frames
	if last_reward_frames > 0:
		last_reward_frames -= 1
		return 0

	pending_reward = obs[pending_reward_idx]
	if pending_reward > 0:
		last_reward_frames = pending_reward + 1
		return pending_reward + 1

	return 0

episode = 0
df = pd.DataFrame(columns=['episode', 'value'])
while episode < 100:
	observation = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		env.render()

		state = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e or counter < hist_size:
			action = np.random.choice(range(n_actions), p=[0.15,0.05,0.20,0.19,0.19,0.22])
			# action = np.random.choice(range(n_actions))
		else:
			sl = list(islice(D, len(D) - (hist_size - 1), len(D)))
			prev_states = [x[0] for x in sl]
			prev_states.append(state)
			stack = np.stack(prev_states, axis=1)
			q_values = model.predict(stack.reshape(1, state_size, hist_size))
			action = q_values[0].argsort()[-1]

		# Take the chosen action
		observation_, reward, done, info = env.step(actions[action])
		# if reward > 0:
		total_catch_value += reward

		reward = get_reward(observation_)

		if reward == 0:
			reward = -0.0001

		# Store the tuple
		state_ = phi(observation_)
		D.append((state, action, reward, state_, done))

		observation = observation_

		# Train the Q function
		if counter > replay_mem_size and not test and counter % update_freq == 0 and len(D) > (batch_size + hist_size):
			D_ = list(D)

			# Train the model
			batch_idxs = np.random.choice(range(hist_size, len(D_)), batch_size)

			X = []
			ys = []
			for i in batch_idxs:
				s, a, r, s_, d = D_[i]

				y = r
				if not d:
					states_ = [x[3] for x in D_[i-(hist_size - 1):i]]
					states_.append(s_)
					stack_ = np.stack(states_, axis=1).reshape(1, state_size, hist_size)
					y = r + gamma * np.amax(model.predict(stack_)[0])

				states = [x[0] for x in D_[i-(hist_size-1):i]]
				states.append(s)
				stack = np.stack(states, axis=1).reshape(1, state_size, hist_size)
				X.append(stack)

				# Calculate the target vector
				target_f = model.predict(stack)
				target_f[0][a] = y
				ys.append(target_f)

			X = np.array(X).reshape(batch_size, state_size, hist_size)
			model.fit(X, np.array(ys).reshape(batch_size, n_actions), epochs=1, verbose=0)

		counter += 1

		if e > e_min and counter > replay_mem_size:
			e -= (1.0 - e_min) / e_decay_frames
			e = max(e_min, e)

	df.loc[episode] = (episode, total_catch_value)
	print('Finished episode', episode, total_catch_value, counter, e)
	df.to_csv('model.csv')

	# if episode % 20 == 0 and not test:
	# 	model_json = model.to_json()
	# 	with open("model.json", "w") as json_file:
	# 		json_file.write(model_json)
	# 	model.save_weights("model.h5")

	episode += 1
