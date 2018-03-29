import gym
from time import time

from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import *
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

from collections import deque
from itertools import islice
import random
import numpy as np
from time import sleep
env = gym.make('FishingDerby-ram-v4')
env.seed(42)

def phi(x):

	line_x = int(x[32])
	line_y = int(x[67])

	fish6_x = int(x[70])
	fish6_y = 245

	v1 = line_x - fish6_x
	v2 = fish6_y - line_y

	v5 = 0 if x[113] == 0 else 1
	return np.array([v1, v2, v5])


observation = env.reset()
state_size = phi(observation).shape[0]

actions = [0,2,3,4,5]
n_actions = 5 #env.action_space.n

print(env.unwrapped.get_action_meanings())
print('State size:', state_size)

test = False
load_model = True

hist_size = 1

# Initialize value function
model = Sequential()
model.add(Flatten(input_shape=(state_size, hist_size)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(n_actions))

print(model.summary())

if load_model:
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")

# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
	error = a - b
	quadratic_term = error*error / 2
	linear_term = abs(error) - 1/2
	use_linear_term = (abs(error) > 1.0)
	if in_keras:
		# Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
		use_linear_term = K.cast(use_linear_term, 'float32')
	return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

opt = RMSprop(lr=0.00025)
model.compile(loss=huber_loss, optimizer=opt)

# Initialize dataset D
D = deque(maxlen=500000)

e = 1.0 if not test else 0.05
e_decay_frames = 100000
e_min = 0.05

gamma = 0.99

update_freq = 32
counter = 0

min_replay_mem_size = 10000
batch_size = 32

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

episode = 0
while True:
	observation = env.reset()

	total_catch_value = 0
	total_value = 0
	done = False
	while not done:
		env.render()

		state = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() < e or counter < hist_size:
			action = np.random.choice(range(n_actions), p=[0.05, 0.24,0.22,0.22,0.27])
			# action = np.random.choice(range(n_actions), p=[0.48,0.52])
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
		if reward > 0:
			total_catch_value += reward

		reward = get_reward(observation, observation_)

		if reward == 0:
			reward = -0.0001

		total_value += reward

		# Store the tuple
		state_ = phi(observation_)
		D.append((state, action, reward, state_, done))

		observation = observation_

		# Train the Q function
		if counter > min_replay_mem_size and not test and counter % update_freq == 0 and len(D) > (batch_size + hist_size):
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
				target_f = np.clip(target_f, -10, 10)
				ys.append(target_f)

			X = np.array(X).reshape(batch_size, state_size, hist_size)
			model.fit(X, np.array(ys).reshape(batch_size, n_actions), epochs=1, verbose=0)

		counter += 1

		if e > e_min and counter > min_replay_mem_size:
			e -= (1.0 - e_min) / e_decay_frames
			e = max(e_min, e)

	print('Finished episode', episode, total_catch_value, total_value, counter, e)

	# if episode % 20 == 0 and not test:
	# 	model_json = model.to_json()
	# 	with open("model.json", "w") as json_file:
	# 		json_file.write(model_json)
	# 	model.save_weights("model.h5")

	episode += 1

