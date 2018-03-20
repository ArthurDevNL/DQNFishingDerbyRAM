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
env = gym.make('BreakoutDeterministic-v4')
env.seed(42)

def phi(x):
	# Convert to grayscale
	x = x[::3,::3]
	x = np.dot(x[...,:3], [0.299, 0.587, 0.114])

	# # Crop image
	# x = x[77:190,28:80]

	# Normalize
	return x

observation = env.reset()
shape = phi(observation).shape
state_shape = phi(observation).shape

actions = [0,1,2,3]
n_actions = 4 #env.action_space.n

hist_size = 4

# s = phi(observation)
# plt.imshow(s, cmap='gray')
# plt.savefig('test')

# Initialize value function
model = Sequential()
model.add(Lambda(lambda x: x / 255.0, input_shape=(hist_size, shape[0], shape[1])))
model.add(Conv2D(16, (8,4), activation='relu'))
model.add(Conv2D(32, (4,2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(n_actions))

opt = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
model.compile(loss='mse', optimizer=opt)

e = 1.0
e_decay_frames = 1000000
e_min = 0.1

gamma = 0.99

update_freq = 4
counter = 0

replay_mem_size = 50000

episode = 0

D = deque(maxlen=1000000)
history = deque(maxlen=4)
while True:
	observation = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		env.render()

		state = phi(observation)
		history.append(state)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e or counter < replay_mem_size:
			action = np.random.choice(range(n_actions))
		else:
			q_values = model.predict(np.array(history).reshape(1, hist_size, shape[0], shape[1]))
			action = q_values[0].argsort()[-1]

		# Take the chosen action
		observation_, reward, done, info = env.step(action)
		reward = np.sign(reward)
		if reward > 0:
			total_catch_value += reward

		# store the data
		state_ = phi(observation_)
		D.append((state, action, reward, state_, done))

		observation = observation_

		# Train the Q function
		if counter > replay_mem_size and counter % update_freq == 0 and len(D) > 32:
			# Train the model
			batch_idxs = np.random.choice(range(hist_size, len(D)), 32)

			X = []
			ys = []
			for i in batch_idxs:
				s, a, r, s_, d = D[i]

				# h = history. Get the previous 3 frames
				h = [D[i-x][0] for x in range(1, hist_size)]



				y = r
				if not d:
					s_ = np.concatenate([s_, s, h[:2]], axis=0).reshape(1, hist_size, shape[0], shape[1])
					y = r + gamma * np.amax(model.predict(s_)[0])

				s = np.concatenate([])
				target_f = model.predict(s.reshape(1, hist_size,state_size))
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

env.close()
