from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras import backend as K
from keras.utils import to_categorical
import numpy as np

import pandas as pd

hist_size=1
state_size=3

actions = [0,2,3,4,5]
n_actions = 5
batch_size = 32
gamma = 0.99

# Initialize value function
model = Sequential()
model.add(Flatten(input_shape=(state_size, hist_size)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(n_actions))

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

# Read data
df = pd.read_csv('data.csv')

for i in range(1000):

	X = []
	ys = []
	for j in range(32):
		row = df.sample(1)
		_, _, s1, s2, s3, a, r, s1_, s2_, s3_ = row.values[0]
		s = np.array([s1, s2, s3])
		s_ = np.array([s1_, s2_, s3_])

		y = r + gamma * np.amax(model.predict(s_.reshape(1, state_size, hist_size))[0])
		target_f = model.predict(s.reshape(1, state_size, hist_size))
		target_f[0][actions.index(a)] = y
		target_f = np.clip(target_f, -10, 10)
		ys.append(target_f)

		X.append(s.reshape(1, state_size, hist_size))

	X = np.array(X).reshape(len(X), state_size, hist_size)
	ys = np.array(ys).reshape(len(ys), n_actions)
	hist = model.fit(X, ys, batch_size=batch_size, epochs=1, verbose=0)

	if i % 10 == 0:
		print(i, hist.history)

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")

# for i in batch_idxs:
# 	s, a, r, s_, d = D_[i]
#
# 	y = r
# 	if not d:
# 		states_ = [x[3] for x in D_[i-(hist_size - 1):i]]
# 		states_.append(s_)
# 		stack_ = np.stack(states_, axis=1).reshape(1, state_size, hist_size)
# 		y = r + gamma * np.amax(model.predict(stack_)[0])
#
# 	states = [x[0] for x in D_[i-(hist_size-1):i]]
# 	states.append(s)
# 	stack = np.stack(states, axis=1).reshape(1, state_size, hist_size)
# 	X.append(stack)
#
# 	# Calculate the target vector
# 	target_f = model.predict(stack)
# 	target_f[0][a] = y
# 	target_f = np.clip(target_f, -10, 10)
# 	ys.append(target_f)

