from baselines.common.vec_env import VecEnvWrapper
import numpy as np
import gym
import cv2

class Goal:
    def __init__(self, rank=0):
        self.rank = rank
        self.walls = None
        self.current_goal = None
        self.past_goals = []
        self.debug = False
    
    # 设置目标点
    def set_goal(self, p, high, low):
        while True:
            dp = np.random.sample(len(high))*(high-low)+ low
            g = (p + dp).astype(np.int)
            if self.is_valid_goal(g):
                break
        self.current_goal = g
    
    # 是否到达目标
    def reach_goal(self, p):
        if np.sum(np.absolute(p.astype(int)-self.current_goal.astype(int))>1)==0:
            return True
        if np.sum(p.astype(int)!=self.current_goal.astype(int))==0:
            return True
        return False

    # 获取可通过区域
    def get_valid_walls(self):
        # empty=46 coin=49 ladder=61 box=35,36,37,38 plant=83,97,98, block=65
        accessable = np.isin(self.walls,[61,35,36,37,38])
        standable = 1-np.isin(self.walls,[46,49])
        # kernel = np.array([[0,0,0],[1,1,0],[0,0,0]],np.uint8)
        kernel = np.array([[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]],np.uint8)
        reachable = cv2.dilate(standable.astype(np.uint8), kernel,iterations=1) - standable + accessable
        if self.debug:
            for i in np.unique(self.walls):
                if not i in [46,49,61,35,36,37,38,83,97,98,65]:
                    cv2.imshow(str(i), np.rot90(cv2.resize((self.walls==i)*1., (300,300), interpolation=cv2.INTER_NEAREST)))
            cv2.imshow('acc', np.rot90(cv2.resize(accessable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
            cv2.imshow('reachable', np.rot90(cv2.resize(reachable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        return reachable
    
    # 判断目标点是否合法
    def is_valid_goal(self, g):
        # 没有越界
        if np.sum(g<0)+ np.sum(g>=self.walls.shape) <= 0:
            # 可通过
            if self.valid_walls[tuple(g)]:
                return True
        return False
    
    def reset(self):
        self.walls = None
        self.current_goal = None
        self.past_goals = []

    def step(self, p, walls):
        if self.walls is None or np.sum(self.walls != walls)>0:
            self.walls = walls
            self.valid_walls = self.get_valid_walls()

        self.p = p.astype(int)
        reward = 0
        done = False
        high = np.array([20, 20])
        low = np.array([-20, -20])
        if self.current_goal is None:
            self.set_goal(p, high, low)
        if self.reach_goal(p):
            reward = 10
            done = True
            self.past_goals.append(self.current_goal)
            self.set_goal(p, high, low)
        return reward, done

    def render(self):
        goals_img = self.valid_walls*0.3
        # goals_img = np.zeros_like(self.walls)
        # print(np.unique(self.walls, return_counts=True))
        for g in self.past_goals:
            goals_img[tuple(g)] = 0.8
        goals_img[tuple(self.current_goal)] = 1
        goals_img[tuple(self.p)] = 1
        # print(self.past_goals)
        cv2.imshow('goals'+str(self.rank), np.rot90(cv2.resize(goals_img, (300,300), interpolation=cv2.INTER_NEAREST)))
        # cv2.imshow('walls'+str(self.rank), np.rot90(cv2.resize(self.valid_walls*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        cv2.waitKey(1)


class CourierWrapper(VecEnvWrapper):
    def __init__(self, venv, debug=False):
        VecEnvWrapper.__init__(self, venv)
        h,w,c = venv.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, shape=[h,w,c+1])
        # init Goal manager
        self.gms = [Goal(i) for i in range(self.num_envs)]
        self.debug = debug

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        pos_channel = np.zeros([self.num_envs, 64, 64, 1])
        walls, ax, ay = self.venv.vec_map_info()
        for i in range(self.num_envs):
            p = np.array([ax[i], ay[i]])
            rews[i], d = self.gms[i].step(p, walls[i])
            if d: dones[i] = d
            pos_channel[i,0,:2,0] = p
            pos_channel[i,0,2:4,0] = self.gms[i].current_goal
            if self.debug:
                self.gms[i].render()
                cv2.imshow('obs', cv2.resize(obs[i],(300,300)))
            if dones[i]:
                self.gms[i].reset()
        self.venv.vec_reset(dones)
        obs = np.concatenate([obs,pos_channel], -1)
        return obs, rews, dones, infos