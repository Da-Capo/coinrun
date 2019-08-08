from baselines.common.vec_env import VecEnvWrapper
import numpy as np
import gym
import cv2

class Goal:
    def __init__(self, rank=0):
        self.rank = rank
        self.n_goals = 4 # 结束条件：需要完成的目标数
        self.reset()
    
    def reset(self):
        self.walls = None        # 地图
        self.current_goal = None # 当前目标位置  
        self.past_goals = []     # 已完成目标

    def step(self, p, walls):
        # 更新 地图 和 可到达位置
        if self.walls is None or (self.walls != walls).any():
            self.walls = walls
            self.valid_walls = self.get_valid_walls()
            self.vp = np.swapaxes(np.where(self.valid_walls>=1),0,1)
        self.p = np.around(p).astype(np.int)
        
        # 处理目标的生成和到达判定
        reward = 0
        if self.current_goal is None:
            self.set_goal()
        elif self.reach_goal(p):
            reward = 1
            self.past_goals.append(self.current_goal)
            self.set_goal()
        done = len(self.past_goals) >= self.n_goals
        
        return reward, done, self.current_goal


    def set_goal(self):
        # 生成随机目标
        self.current_goal = self.vp[np.random.randint(self.vp.shape[0])]
    
    def reach_goal(self, p):
        # 到达判定
        if (np.absolute(p-self.current_goal)<=1).all():
            return True
        return False

    def get_valid_walls(self):
        # 可到达区域计算
        # empty=46 coin=49 ladder=61 box=35,36,37,38 plant=83,97,98, block=65
        accessable = np.isin(self.walls,[61,35,36,37,38])
        standable = 1-np.isin(self.walls,[46,49])
        kernel = np.array([[1]*7+[1]+[0]*7],np.uint8)
        reachable = cv2.dilate(standable.astype(np.uint8), kernel,iterations=1) - standable + accessable
        ######### debug info ##########
        # for i in np.unique(self.walls):
        #     if not i in [46,49,61,35,36,37,38,83,97,98,65]:
        #         cv2.imshow(str(i), np.rot90(cv2.resize((self.walls==i)*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        # cv2.imshow('acc', np.rot90(cv2.resize(accessable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        # cv2.imshow('reachable', np.rot90(cv2.resize(reachable*1., (300,300), interpolation=cv2.INTER_NEAREST)))
        return reachable

    def render(self, walls=None):
        if walls==None:walls=self.walls
        goals_img = walls/255.
        for g in self.past_goals:
            goals_img[tuple(g)] = 0.8
        goals_img[tuple(self.current_goal)] = 1
        goals_img[tuple(self.p)] = 1
        cv2.imshow('goals'+str(self.rank), np.rot90(cv2.resize(goals_img, (300,300), interpolation=cv2.INTER_NEAREST)))
        cv2.waitKey(1)

class CourierWrapper(VecEnvWrapper):
    def __init__(self, venv, goal_fn=Goal, input_pixel=False, debug=False):
        VecEnvWrapper.__init__(self, venv)
        self.size = h,w,c = (15,15,1)
        self.observation_space = gym.spaces.Box(0, 255, shape=[h,w,c+1], dtype=np.float32)

        # init Goal manager
        self.gms = [goal_fn(i) for i in range(self.num_envs)]
        self.debug = debug
        self.input_pixel = input_pixel

    def reset(self):
        obs, _, _, _ = self.step_wait()
        return obs

    def step_wait(self):
        oobs, rews, dones, infos = self.venv.step_wait()
        # walls 全地图信息
        # mwalls 包含怪物的全地图信息
        obs_p, walls, mwalls = self.prepare_obs()

        # oobs 原始的像素图
        # obs_p 包含 obs 和 坐标 两个通道
        if self.input_pixel:
            obs_p[...,0] = oobs
        
        
        ds = []
        for i in range(self.num_envs):
            if dones[i]:
                self.gms[i].reset()
            # 计算出目标奖励
            rews[i], d, t = self.gms[i].step(obs_p[i,0,:2,1], walls[i])
            # 通道1中存入人物&目标的坐标信息
            obs_p[i,0,2:4,1] = t
            # 加入速度信息
            # obs_p[i,:2,:2, 0] = oobs[i, :2, :2]
            # obs_p[i,:2,2:4, 0] = oobs[i, :2, 22:24]

            ds.append(d)

            if self.debug:
                self.gms[i].render()
                cv2.imshow('oobs', cv2.resize(oobs[i]/255.,(300,300), interpolation=cv2.INTER_NEAREST))
                cv2.imshow('obs', cv2.resize(obs_p[i,:,:,0]/255.,(300,300), interpolation=cv2.INTER_NEAREST))
        
        # 主动done掉某个环境    
        self.venv.vec_terminate(ds)
        return obs_p, rews, dones, infos
    
    def prepare_obs(self, is_monster=True):
        # 获取地图信息拼接成obs
        obs = np.zeros([self.num_envs, self.size[0], self.size[1], 2])
        walls, ax, ay = self.venv.vec_map_info()
        ax = ax.astype(int)
        ay = ay.astype(int)
        mx = self.venv.buf_mx.reshape(-1, 20).astype(int)
        my = self.venv.buf_my.reshape(-1, 20).astype(int)
        mwalls = walls.copy()
        for i in range(self.num_envs):
            # 绘制怪物 & 人物
            for x,y in zip(mx[i], my[i]):
                if x!= 0 or y!=0:
                    mwalls[i, x, y] = 150
            mwalls[i, ax[i], ay[i]] = 200
            
            # 获取可视区域
            pmwalls = np.pad(mwalls[i],int((self.size[0]-1)/2), 'constant')

            obs[i,:,:,0] = np.rot90(pmwalls[ax[i]:(ax[i]+self.size[0]), ay[i]:(ay[i]+self.size[1])])
            obs[i,:,:2,1] = np.array([ax[i], ay[i]])
        return obs, walls, mwalls

    def test_gms(self):
        p = np.zeros(2)
        while True:
            self.step(np.array([0]))
            walls,_,_ = self.venv.vec_map_info()
            r, d, t = self.gms[0].step(p, walls[0])
            # print(d, p, self.gms[0].current_goal)
            p = self.gms[0].current_goal
            self.venv.vec_terminate([d])
            if d:
                print(self.gms[0].past_goals)

if __name__=="__main__":
    import sys
    sys.path.append("..")
    from coinrun import setup_utils, make
    setup_utils.setup_and_load(use_cmd_line_args=False, use_black_white=True, num_levels=-1)
    env = CourierWrapper(make('platform', num_envs=1))
    env.test_gms()