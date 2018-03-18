import gym
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import *
from collections import deque
from itertools import islice
import numpy as np
from time import sleep
env = gym.make('Breakout-ram-v0')
env.seed(42)

# Logging and monitoring
# logger.set_level(logger.INFO)
# outdir = 'monitor/results'
# env = wrappers.Monitor(env, directory=outdir, force=True)

test = False
load_model = test

observation = env.reset()
state_size = observation.shape[0]

actions = [0,1,2,3]#,3,4,5]
n_actions = 4 #env.action_space.n

print(env.unwrapped.get_action_meanings())

hist_size = 4

# Initialize value function
model = Sequential()
model.add(Flatten(input_shape=(state_size, hist_size)))
model.add(Lambda(lambda x: x / 255.0))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(n_actions))

opt = RMSprop(lr=0.0001, rho=0.95, epsilon=0.01)
model.compile(loss='mse', optimizer=opt)

# Initialize dataset D
D = deque(maxlen=100000)

e = 1.0 if not test else 0.1
e_decay_frames = 1000000
e_min = 0.1

gamma = 0.99

update_freq = 4
counter = 0

# replay_mem_size = 50000
replay_mem_size = 1000
batch_size = 32

episode = 0
while True:
	state = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		env.render()

		if test:
			sleep(0.01)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e or counter < replay_mem_size:
			action = np.random.choice(range(n_actions))
		else:
			sl = list(islice(D, len(D) - (hist_size - 1), len(D)))
			prev_states = [x[0] for x in sl]
			prev_states.append(state)
			stack = np.stack(prev_states, axis=1)
			q_values = model.predict(stack.reshape(1, state_size, hist_size))
			action = q_values[0].argsort()[-1]

			if test:
				print(action)

		# Take the chosen action
		state_, reward, done, info = env.step(actions[action])
		if reward > 0:
			total_catch_value += reward

		# reward = max(0, reward)
		if reward == 0:
			reward = -0.01

		# #
		# if reward > 0:
		# 	reward *= 2

		# Store the tuple
		D.append((state, action, reward, state_, done))

		state = state_

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

	print('Finished episode', episode, total_catch_value, counter, e)

	# if episode % 20 == 0:
	# 	model_json = model.to_json()
	# 	with open("model.json", "w") as json_file:
	# 		json_file.write(model_json)
	# 	model.save_weights("model.h5")

	episode += 1
