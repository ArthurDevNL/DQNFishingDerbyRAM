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

test = True
load_model = True

hist_size = 1

# Initialize value function
model = None
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


e = 0.05

gamma = 0.99

counter = 0

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
			q_values = model.predict(state.reshape(1, state_size, hist_size))
			action = q_values[0].argsort()[-1]

		# Take the chosen action
		observation_, reward, done, info = env.step(actions[action])

		# Store the tuple
		state_ = phi(observation_)

		observation = observation_

	print('Finished episode', episode, total_catch_value, total_value, counter, e)

	episode += 1

