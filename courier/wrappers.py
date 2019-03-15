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
        self.level = 0
    
    def reset(self):
        self.walls = None
        self.current_goal = None
        self.past_goals = []
        self.level = 0

    def step(self, p, walls):
        if self.walls is None or (self.walls != walls).any():
            self.walls = walls
            self.valid_walls = self.get_valid_walls()
        
        self.p = np.around(p).astype(np.int)
        reward = 0
        done = False
        if self.current_goal is None:
            self.set_goal()
        if self.reach_goal(p):
            reward = 1
            self.level += 1
            if self.level > 6:
                done = True
            self.past_goals.append(self.current_goal)
            self.set_goal()
        return reward, done


    def set_goal(self):
        # high = np.array([20, 20])
        # low = np.array([-20, -20])
        high = np.array([self.walls.shape[0], 5*(self.level+1)])
        low = np.array([0, 5*self.level])
        while True:
            g = np.around(np.random.sample(len(high))*(high-low)+ low).astype(np.int)
            if self.is_valid_goal(g):
                break
        self.current_goal = g
    
    def reach_goal(self, p):
        if (np.absolute(p-self.current_goal)<=1).all():
            return True
        return False

    def get_valid_walls(self):
        # empty=46 coin=49 ladder=61 box=35,36,37,38 plant=83,97,98, block=65
        accessable = np.isin(self.walls,[61,35,36,37,38])
        standable = 1-np.isin(self.walls,[46,49])
        kernel = np.array([[1]*7+[1]+[0]*7],np.uint8)
        reachable = cv2.dilate(standable.astype(np.uint8), kernel,iterations=1) - standable + accessable
        if False:
            for i in np.unique(self.walls):
                if not i in [46,49,61,35,36,37,38,83,97,98,65]:
                    cv2.imshow(str(i), np.rot90(cv2.resize((self.walls==i)*1., (300,300), interpolation=cv2.INTER_NEAREST)))
            cv2.imshow('acc', np.rot90(cv2.resize(accessable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
            cv2.imshow('reachable', np.rot90(cv2.resize(reachable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        return reachable
    
    def is_valid_goal(self, g):
        if (g>0).all() and (g<self.walls.shape).all():
            if self.valid_walls[tuple(g)]:
                return True
        return False
    
    def render(self):
        goals_img = self.valid_walls*0.3
        for g in self.past_goals:
            goals_img[tuple(g)] = 0.8
        goals_img[tuple(self.current_goal)] = 1
        goals_img[tuple(self.p)] = 1
        cv2.imshow('goals'+str(self.rank), np.rot90(cv2.resize(goals_img, (300,300), interpolation=cv2.INTER_NEAREST)))
        cv2.waitKey(1)


class CourierWrapper(VecEnvWrapper):
    def __init__(self, venv, goal_fn=Goal, debug=False):
        VecEnvWrapper.__init__(self, venv)
        h,w,c = venv.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, shape=[h,w,c+1])

        # init Goal manager
        self.gms = [goal_fn(i) for i in range(self.num_envs)]
        self.debug = debug

    def reset(self):
        obs, _, _, _ = self.step_wait()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        pos_channel = np.zeros([self.num_envs, 64, 64, 1])
        walls, ax, ay = self.venv.vec_map_info()
        ds = []
        for i in range(self.num_envs):
            p = np.array([ax[i], ay[i]])
            rews[i], d = self.gms[i].step(p, walls[i])
            ds.append(d)
            # if d: dones[i] = d
            pos_channel[i,0,:2,0] = p
            pos_channel[i,0,2:4,0] = self.gms[i].current_goal
            if self.debug:
                self.gms[i].render()
                cv2.imshow('obs', cv2.resize(obs[i],(300,300)))
            if dones[i]:
                self.gms[i].reset()
            
        self.venv.vec_terminate(ds)
        obs = np.concatenate([obs,pos_channel], -1)
        return obs, rews, dones, infos