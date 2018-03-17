import gym
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from collections import deque
import numpy as np
from time import sleep
env = gym.make('FishingDerby-ram-v4')
env.seed(42)

# Logging and monitoring
# logger.set_level(logger.INFO)
# outdir = 'monitor/results'
# env = wrappers.Monitor(env, directory=outdir, force=True)

test = False
load_model = test

observation = env.reset()
state_size = observation.shape[0]

actions = [0,1,2,3,4,5]
n_actions = 6 #env.action_space.n

# Initialize dataset D
D = deque(maxlen=100000)

hist_size = 4

# Initialize value function
model = Sequential()
model.add(Flatten(input_shape=(hist_size, state_size)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_actions))

opt = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
model.compile(loss='mse', optimizer=opt)

def phi(x):
	return x / 255.0

e = 1.0 if not test else 0.1
e_decay_frames = 1000000
e_min = 0.1

gamma = 0.99

update_freq = 4
counter = 0

replay_mem_size = 50000

episode = 0

history = deque(maxlen=hist_size)
while True:
	observation = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		env.render()

		if test:
			sleep(0.01)

		state = phi(observation)
		history.append(state)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e or counter < replay_mem_size:
			action = np.random.choice(range(n_actions))
			# action = env.action_space.sample()
		else:
			q_values = model.predict(np.array(history).reshape(1, hist_size, state_size))
			action = q_values[0].argsort()[-1]

			if test:
				print(action)

		# Take the chosen action
		observation_, reward, done, info = env.step(action)
		if reward > 0:
			total_catch_value += reward

		reward = max(0, reward)
		# if reward == 0:
		# 	reward = -0.01
		# #
		# if reward > 0:
		# 	reward *= 2

		# Store the tuple
		state_ = phi(observation_)
		if len(history) == 4:
			D.append((np.array(history).reshape(hist_size, state_size), action, reward, state_, done))

		observation = observation_

		# Train the Q function
		if counter > replay_mem_size and not test and counter % update_freq == 0 and len(D) > 32:
			# Train the model
			batch_idxs = np.random.choice(range(len(D)), 32)
			batch = [D[i] for i in batch_idxs]

			X = []
			ys = []
			for s, a, r, s_, d in batch:

				y = r
				if not d:
					s_ = np.concatenate([s,s_.reshape(1,128)], axis=0)[1:].reshape(1, hist_size, state_size)
					y = r + gamma * np.amax(model.predict(s_)[0])

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

	if episode % 20 == 0:
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")

	episode += 1
