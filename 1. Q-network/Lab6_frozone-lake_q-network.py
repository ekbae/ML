# -*- coding: utf-8 -*-
# code from: https://www.inflearn.com/course/%EA%B8%B0%EB%B3%B8%EC%A0%81%EC%9D%B8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B0%95%EC%A2%8C/

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x):
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v0')     # stochastic world

#input, output from gym
input_size = env.observation_space.n    # 16
output_size = env.action_space.n        # 4
learning_rate = 0.1

tf.reset_default_graph()
#These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1,input_size],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size,output_size],0,0.01))

Qpred = tf.matmul(X,W)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)    #nextQ
loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#Set Q-learning related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total reward and steps per episode
rList = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 초기화
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False

        # The Q-Network training
        while not done:
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})

            # Choose an action by greedily (with a chane of random action) from the Q-network
            # e 보다 rand 값이 작으면, action_space 그대로 : decay E-greedy
            if np.random.rand(1)<e:
                a = env.action_space.sample()
            else:
                a= np.argmax(Qs)

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a)

            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, a] = reward   # Qs[a]가 아닌이유: 1*4 의 array 이기 때문
            else:
                Qs1 = sess.run(Qpred, feed_dict={X:one_hot(s1)})
                #update Q
                Qs[0, a] = reward + dis * np.max(Qs1)
            ## a만 update 함

            # Train our network using Target * Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

            rAll += reward
            s=s1
        rList.append(rAll)

# train our network using target (Y) and predicted Q

print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()

