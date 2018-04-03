from cartpole import CartPoleEnv

env = CartPoleEnv(length=1.0)

env.reset()

for step in range(1000):
	action = 0
	next_state,reward,done,_ = env.step(0)

	if done:
		print "done reward:",reward
		break