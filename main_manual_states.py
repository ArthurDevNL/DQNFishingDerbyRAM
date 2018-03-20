import gym
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import *
from keras.utils import to_categorical
from collections import deque
from itertools import islice
import numpy as np
from time import sleep
env = gym.make('FishingDerby-ram-v0')
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
	fishes = [69, 70, 71, 72, 73, 74]
	for f in fishes:
		v = rescale(x[f], min_x, max_x)
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
	v = rescale(x[line_y], 200, 252)
	features.append(v)

	caught_fish_idx = 112
	v = 0 if x[caught_fish_idx] == 0 else 1
	features.append(v)

	return np.array(features)

observation = env.reset()
state_size = phi(observation).shape[0]

actions = [0,1,2,3,4,5]#,3,4,5]
n_actions = 6 #env.action_space.n

print(env.unwrapped.get_action_meanings())
print('State size:', state_size)

# Initialize value function
model = Sequential()
model.add(Dense(32, input_dim=state_size, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(n_actions))

opt = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=opt)

# Initialize dataset D
D = deque(maxlen=100000)

e = 1.0
e_decay_frames = 500000
e_min = 0.1

gamma = 0.99

update_freq = 4
counter = 0

replay_mem_size = 30000
batch_size = 32

episode = 0
while True:
	observation = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		# env.render()

		state = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e or counter < replay_mem_size:
			action = np.random.choice(range(n_actions))
		else:
			q_values = model.predict(state.reshape(1,state_size))
			action = q_values[0].argsort()[-1]

		# Take the chosen action
		observation_, reward, done, info = env.step(actions[action])
		if reward > 0:
			total_catch_value += reward

		reward = max(0, reward)
		if reward == 0:
			reward = -0.1

		# Store the tuple
		state_ = phi(observation_)
		D.append((state, action, reward, state_, done))

		observation = observation_

		# Train the Q function
		if counter > replay_mem_size and counter % update_freq == 0 and len(D) > batch_size:
			# Train the model
			batch_idxs = np.random.choice(range(len(D)), 32)
			batch = [D[i] for i in batch_idxs]

			X = []
			ys = []
			for s, a, r, s_, d in batch:

				y = r
				if not d:
					y = r + gamma * np.amax(model.predict(s_.reshape(1,state_size))[0])

				target_f = model.predict(s.reshape(1,state_size))
				target_f[0][a] = y
				ys.append(target_f)
				X.append(s)

			X = np.array(X)
			model.fit(X, np.array(ys).reshape(32, n_actions), epochs=1, verbose=0)

		counter += 1

		if e > e_min and counter > replay_mem_size:
			e -= (1.0 - e_min) / e_decay_frames
			e = max(e_min, e)

	print('Finished episode', episode, total_catch_value, counter, e)

	if episode % 20 == 0:
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")

	episode += 1
