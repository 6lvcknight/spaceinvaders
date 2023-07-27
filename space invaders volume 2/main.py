import gym
import random

env = gym.make("ALE/SpaceInvaders-v5", render_mode = 'human')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# Specify the desired render mode and fps
render_mode = 'human'
render_fps = 30  # Set the desired fps value (e.g., 30)

env.unwrapped.get_action_meanings()

episodes = 5
for e in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # Render the environment with the specified render mode and fps
        env.render()

        action = random.choice([0, 1, 2, 3, 4, 5])
        result = env.step(action)
        score += result[1]

    print('Episode:{} Score:{}'.format(e, score))

env.close()

'''creating a deep learning model with keras'''

import numpy as np
import tensorflow as tf


def build_model(height, width, channels, actions):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Convolution2D(32, (8,8), strides = (4,4), activation = 'relu', input_shape = (3, height, width, channels)))
    model.add(tf.keras.layers.Convolution2D(64, (4,4), strides = (2,2), activation = 'relu'))
    model.add(tf.keras.layers.Convolution2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(256, activation = 'relu'))
    model.add(tf.keras.layers.Dense(actions, activation = 'linear'))
    return model

model = build_model(height, width, channels, actions)

model.summary()

'''build agent with keras-RL'''

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length = 3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True,
                   dueling_type='avg', nb_actions=actions, nb_steps_warmups=1000)
    return dqn

dqn = build_agent(model, actions)
dqn = compile(tf.keras.optimizers.Adam(lr=1e-4))
dqn.fit(env, nb_steps=10000 , visualize=False, verbose=2) #training steps

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episodes_reward']))

'''reloading agent from memory'''

dqn.save_weights('Savedweights/10k-Fast/dqn_weights.h5f')
dqn.load_weights('Savedweights/1m/dqn_weights.h5f')