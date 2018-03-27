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

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# mn = min, mx = max
def rescale(v, mn, mx):
	v = min(v - mn, mx - mn)
	return float(v / (mx - mn))

def one_hot(x, mn, mx, num_classes=20):
	x = max(x - mn, 0)
	x = min(x, mx - mn)
	x = int(round(x/num_classes))
	return to_categorical(x, num_classes=num_classes)

def phi(x):

	features = []

	line_x = int(x[32])
	line_y = int(x[67])

	fish2_top_x = int(x[74])
	fish4_top_x = int(x[72])
	fish6_top_x = int(x[70])

	# v1 = max(line_y - 207, 0) # or
	# v1 = 245 - line_y
	# v1_1 = 245 - line_y

	# Distance to fish 4
	xclip = 20
	v2 = fish2_top_x - line_x
	v2y = 217 - line_y
	v3 = fish4_top_x - line_x
	v3y = 230 - line_y
	v4 = fish6_top_x - line_x
	v4y = 245 - line_y

	shark_x = int(x[75])
	# shark_y = 213
	v5 = shark_x - line_x + 5

	# v4 = shark_y - line_y
	# v4 = np.clip([v4], -20, 20)[0]

	caught_fish_idx = 112
	v0 = int(x[caught_fish_idx])
	return np.array([v0, v2, v2y, v3, v3y, v4, v4y])

observation = env.reset()
state_size = phi(observation).shape[0]

actions = [2,3,4,5]
n_actions = 4 #env.action_space.n

print(env.unwrapped.get_action_meanings())
print('State size:', state_size)

test = False
load_model = False

hist_size = 1

# Initialize value function
model = Sequential()
model.add(Flatten(input_shape=(state_size, hist_size)))
model.add(Dense(16))
model.add(Dense(16))
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
e_decay_frames = 200000
e_min = 0.1

gamma = 0.99

update_freq = 32
counter = 0

min_replay_mem_size = 32
batch_size = 32

pending_reward_idx = 114
last_reward_frames = 0
caught_fish_idx = 112
def get_reward(obs, obs_):

	# if obs_[caught_fish_idx] == 0 and obs[caught_fish_idx] > 0 and obs_[pending_reward_idx] == 0:
	# 	return -1

	global last_reward_frames
	if last_reward_frames > 0:
		last_reward_frames -= 1
		return 0

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
			action = np.random.choice(range(n_actions), p=[0.26,0.23,0.23,0.28])
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
			reward = -0.01

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

	if episode % 20 == 0 and not test:
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")

	episode += 1

