# -*- coding: utf-8 -*-
# code from: https://www.inflearn.com/course/%EA%B8%B0%EB%B3%B8%EC%A0%81%EC%9D%B8-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B0%95%EC%A2%8C/

# NIPS 2013 논문은
# 아직 target Q를 분리하지 않았을 때의 알고리즘이다.

import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque
import gym

env = gym.make('CartPole-v0') # https://github.com/openai/gym/wiki/CartPole-v0
env._max_episode_steps = 10004 # cartPole has terminal conditions like "Episode length is greater than 200"

#Constants defining our neural network
input_size = env.observation_space.shape[0] # 4개 action: Cart Position, Cart Velocity, Pole angle, Pole velocity at tip
output_size = env.action_space.n            # 2개 action: Push cart to the left/right

#termianl condition: Pole angle +-12 degree, Cart Position +-2.4, Episode lendgth over 10004


dis = 0.9
REPLAY_MEMORY = 50000

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0,DQN.input_size)
    y_stack = np.empty(0).reshape(0,DQN.output_size)

    #Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q=DQN.predict(state)

        if done:        #terminal?
            Q[0, action] = reward
        else: # obtain the Q' value by feeding the new state through our network
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

        #train our network using target and predicted Q values on each episode
    return DQN.update(x_stack, y_stack)

def bot_play(mainDQN):
    #see our trained network in action
    s= env.reset()
    reward_sum = 0
    while True:
        env.render()
        a= np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    max_episodes = 5000
    #store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, "main")
        tf.global_variables_initializer().run()

        for episode in range(max_episodes):
            e=1./((episode /10) +1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                #Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                if done:    #big penalty
                    reward = -100

                #save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000: #good enough
                    break

            print ("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000:
                pass

            if episode %10 == 1: #train every 10 epiosodes. get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch)
                print ("Loss: ", loss)

        bot_play(mainDQN)

if __name__ == '__main__':
    main()