import gym
import numpy as np
import pickle
import pandas as pd

env = gym.make('FishingDerby-ram-v4')
env.seed(42)

test = True

def bucket(v, mx, buckets=10):
	v = min(v, mx)
	return round(float(v / mx) * float(buckets))

def phi(x):

	line_x = int(x[32])
	line_y = int(x[67])
	fish6_top_x = int(x[70])

	x_dist = fish6_top_x - line_x

	xleft = abs(x_dist) if x_dist < 0 else 0
	xright = x_dist if x_dist > 0 else 0

	y_dist = 245 - line_y
	ytop = abs(y_dist) if y_dist < 0 else 0
	ybot = y_dist if y_dist > 0 else 0

	caught_fish_idx = 112
	v0 = 0 if x[caught_fish_idx] != 2 else 1

	res = np.clip([xleft, xright, ytop, ybot], 0, 20)
	return (res[0], res[1], res[2], res[3], v0)

observation = env.reset()
# state_size = phi(observation).shape[0]
state_size = len(phi(observation))

actions = [2,3,4,5]#,3,4,5]
n_actions = 4 #env.action_space.n

print(env.unwrapped.get_action_meanings())
print('State size:', state_size)

e = 1.0
e_decay_frames = 100000
e_min = 0.05

alpha = 0.1
gamma = 0.99

counter = 0

pending_reward_idx = 114
last_reward_frames = 0
caught_fish_idx = 112
def get_reward(obs, obs_):

	# if obs_[caught_fish_idx] == 0 and obs[caught_fish_idx] == 6 and obs_[pending_reward_idx] == 0:
	# 	return -0.5

	global last_reward_frames
	if last_reward_frames > 0:
		last_reward_frames -= 1
		return 0

	if obs_[caught_fish_idx] == 2 and obs_[67] <= 210:
		pending_reward = abs(obs_[caught_fish_idx] - 7)
		last_reward_frames = pending_reward + 1
		return pending_reward + 1

	return 0

Q = {}
if test:
	u = pickle._Unpickler(open("Q.dump", "rb"))
	u.encoding = 'latin1'
	Q = u.load()

def getQ(s, a):
	if (s,a) not in Q:
		return 0.0
	else:
		return Q[(s,a)]

def learnQ(state, action, reward, value):
	v = getQ(state, action)
	if v is None:
		Q[(state, action)] = reward
	else:
		Q[(state, action)] = v + alpha * (value - v)

def learn(state1, action1, reward, state2):
	maxqnew = max([getQ(state2, a) for a in actions])
	learnQ(state1, action1, reward, reward + gamma * maxqnew)

episode = 0
df = pd.DataFrame(columns=['episode', 'value'])
while episode < 100:
	observation = env.reset()

	total_catch_value = 0
	total_value = 0
	done = False
	while not done:
		# env.render()

		state = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = np.random.choice(range(n_actions), p=[0.26,0.23,0.23,0.28])
		# action = np.random.choice(range(n_actions))

		# Take the chosen action
		observation_, reward, done, info = env.step(actions[action])
		# if reward > 0:
		total_catch_value += reward

		reward = get_reward(observation, observation_)

		# if reward == 0:
		# 	reward = -0.01

		total_value += reward

		# Store the tuple
		state_ = phi(observation_)
		# if not test:
		# 	learn(state, action, reward, state_)

		observation = observation_

		counter += 1

		# Anneal epsilon
		if e > e_min and not test:
			e -= (1.0 - e_min) / e_decay_frames
			e = max(e_min, e)


	df.loc[episode,:] = (episode, total_catch_value)
	df.to_csv('nonuniform_random_100.csv')
	print('Finished episode', episode, total_catch_value, total_value, counter, e)

	episode += 1

df.to_csv('nonuniform_random_100.csv')