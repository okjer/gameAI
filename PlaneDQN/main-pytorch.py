# coding:UTF-8

from __future__ import print_function
import rl_brain_pytorch
import cv2
import numpy as np
import datetime
import tensorflow as tf
import sys
sys.path.append("game/")
import plane as game

TRAINING = False

# [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
def preprocess(observation, reshape):
    "将游戏画面转换成黑白并且调整图片大小"
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    if reshape:
        return np.reshape(observation, (1,80, 80))
    else:
        return observation

def main():
    begin_time = datetime.datetime.now()

    env = game.GameState()
    #env.display = ~TRAINING
    brain = rl_brain_pytorch.DeepQNetwork()

    step = 0
    for episode in range(rl_brain_pytorch.MAX_EPISODE):
        # do nothing
        observation, _, _ = env.frame_step([1,0,0])
        observation = preprocess(observation, False)
        brain.reset(observation)
        score = 0.0
        while True:
            action = brain.choose_action(observation)
            observation_, reward, done = env.frame_step(action)
            if reward == 1: score+=1
            observation_ = preprocess(observation_, True)
            if TRAINING:
                brain.store_transition(observation, action, reward, done, observation_)
            # 有一定的记忆就可以开始学习了
            if step > 200:
               if TRAINING:
                    brain.learn()

            if done:
                break

            observation = observation_
            step += 1
            
        end_time = datetime.datetime.now()
        print("episode {} over. exec time:{} step:{} score:{}".format(episode, end_time - begin_time, step,score))
    brain.saveNet()
    env.exit("game over")


if __name__ == "__main__":
    #if len(sys.argv) == 2 and sys.argv[1] == 'train':
    #r    INITIAL_EPSILON = 0.1
    main()
