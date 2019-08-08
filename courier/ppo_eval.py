from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.vec_env.vec_monitor import VecMonitor
import sys
sys.path.append("..")
from coinrun import setup_utils, make
from wrappers import CourierWrapper
from ppo_train import goal_network

setup_utils.setup_and_load(use_cmd_line_args=False, use_black_white=True, paint_vel_info=1, num_levels=1,set_seed=3)
env = VecMonitor(CourierWrapper(make('platform', num_envs=96)))

model = ppo2.learn(env=env, 
                   network=goal_network,
                   total_timesteps=int(0),
                   log_interval=10,
                    load_path="baselinesLog/ppo/..."
                  )

obs = env.reset()
while True:
    a = model.step(obs)[0]
    obs = env.step(a)[0]
    env.render()