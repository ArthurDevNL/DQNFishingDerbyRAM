import gym
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import *
from collections import deque
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from time import sleep
env = gym.make('FishingDerby-v0')
env.seed(42)

def phi(x):
	# Convert to grayscale
	x = np.dot(x[...,:3], [0.299, 0.587, 0.114])

	# Crop image
	x = x[77:190,28:80]

	# Normalize
	return x / 255.0

observation = env.reset()
state_shape = phi(observation).shape

actions = [2,3,4,5]
n_actions = 4 #env.action_space.n

# Initialize dataset D
D = deque(maxlen=10000)

# Initialize value function
model = Sequential()
model.add(Conv2D(16, (8,4), input_shape=(state_shape[0], state_shape[1], 1), activation='relu'))
model.add(Conv2D(32, (4,2), activation='relu'))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(n_actions))

opt = RMSprop(lr=-.0001)
model.compile(loss='mse', optimizer=opt)

e = 1.0
e_decay = 0.99
e_min = 0.01

gamma = 0.95

update_freq = 100
counter = 0

for episode in range(1000):
	observation = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		env.render()

		state = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e:
			action = np.random.choice(range(n_actions))
			# action = env.action_space.sample()
		else:
			q_values = model.predict(state.reshape(1,state_shape[0], state_shape[1]))
			action = q_values[0].argsort()[-1]

		# Take the chosen action
		observation_, reward, done, info = env.step(action + 2)
		if reward > 0:
			total_catch_value += reward

		# reward = max(0, reward)
		# if reward == 0:
		# 	reward = -0.001
		#
		# if reward > 0:
		# 	reward *= 2

		# Store the tuple
		state_ = phi(observation_)
		D.append((state, action, reward, state_, done))

		observation = observation_

		# Train the Q function
		if counter % update_freq == 0 and len(D) > 32:
			# Train the model
			batch_idxs = np.random.choice(range(len(D)), 32)
			batch = [D[i] for i in batch_idxs]
			for s, a, r, s_, d in batch:

				y = r
				if not d:
					y = r + gamma * np.amax(model.predict(s_.reshape(1,state_shape[0], state_shape[1],1))[0])

				# target_f = model.predict(s.reshape(1,state_size))
				target_f = np.zeros((1,4))
				target_f[0][a] = y
				model.fit(s.reshape(1,state_shape[0], state_shape[1],1), target_f, epochs=1, verbose=0)

		counter += 1

	print('Finished episode', episode, total_catch_value, e)

	if e > e_min:
		e *= e_decay

env.close()
