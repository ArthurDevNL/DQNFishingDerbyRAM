import gym
import numpy as np
env = gym.make('FishingDerby-ram-v4')
env.seed(42)

def phi(x):

	features = []

	line_x = int(x[32])
	line_y = int(x[67])

	fish4_top_x = int(x[72])
	fish4_top_y = 230

	# Distance to fish 4
	v1 = fish4_top_x - line_x
	v2 = fish4_top_y - line_y

	fish6_x = int(x[70])
	fish6_y = 245
	# features.append(fish6_x - line_x)
	# features.append(fish6_y - line_y)

	shark_x = int(x[75])
	shark_y = 213
	v3 = shark_x - line_x
	v4 = shark_y - line_y

	caught_fish_idx = 112
	v5 = 0 if x[caught_fish_idx] == 0 else 1

	return (v1, v2, v3, v4, v5)

observation = env.reset()
# state_size = phi(observation).shape[0]
state_size = len(phi(observation))

actions = [0,1,2,3,4,5]#,3,4,5]
n_actions = 6 #env.action_space.n

print(env.unwrapped.get_action_meanings())
print('State size:', state_size)

test = False
load_model = False

e = 1.0 if not test else 0.05
e_decay_frames = 1000000
e_min = 0.1

gamma = 0.95

counter = 0

pending_reward_idx = 114
last_reward_frames = 0
def get_reward(obs):
	global last_reward_frames
	if last_reward_frames > 0:
		last_reward_frames -= 1
		return 0

	pending_reward = obs[pending_reward_idx]
	if pending_reward > 0:
		last_reward_frames = pending_reward + 1
		return pending_reward + 1

	return 0


Q = {}

alpha = 0.1

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
while True:
	observation = env.reset()

	total_catch_value = 0
	done = False
	while not done:
		# env.render()

		state = phi(observation)

		# Take a random action fraction e (epsilon) of the time
		action = None
		if np.random.rand() <= e:
			# action = np.random.choice(range(n_actions), p=[0.10,0.05,0.22,0.19,0.19,0.25])
			action = np.random.choice(range(n_actions))
		else:
			q = [getQ(state, a) for a in actions]
			maxQ = np.argmax(q)
			action = actions[maxQ]

		# Take the chosen action
		observation_, reward, done, info = env.step(actions[action])
		if reward > 0:
			total_catch_value += reward

		reward = get_reward(observation_)

		if reward == 0:
			reward = -0.001

		# Store the tuple
		state_ = phi(observation_)

		observation = observation_

		counter += 1

		# Anneal epsilon
		if e > e_min:
			e -= (1.0 - e_min) / e_decay_frames
			e = max(e_min, e)

	print('Finished episode', episode, total_catch_value, counter, e)

	episode += 1

