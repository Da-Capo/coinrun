from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.vec_env.vec_monitor import VecMonitor
import datetime
import sys
sys.path.append("..")
from coinrun import setup_utils, make
from wrappers import CourierWrapper



logdir = 'Log/'+datetime.datetime.now().strftime('baselines-%Y-%m-%d/%H%M%S%f')
logger.configure(logdir,['stdout','log', 'tensorboard'])

setup_utils.setup_and_load(use_cmd_line_args=False, use_black_white=True, paint_vel_info=1, num_levels=1,set_seed=3)
env = VecMonitor(CourierWrapper(make('platform', num_envs=96)))

def goal_network(x, **conv_kwargs):
    from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
    import numpy as np
    import tensorflow as tf
    frame = x[:,:,:,:-1]/255.
    pg = x[:,0,:4,-1]/64.
    # print('-------------------------',x,frame,p,g)
    def activ(curr):
        return tf.nn.relu(curr)
    h = activ(conv(frame, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    hpg = activ(fc(tf.concat([p,g], axis=1), 'fc1', nh=128, init_scale=np.sqrt(2)))
    h3 = tf.concat([h3,hpg], axis=1)
    return activ(fc(h3, 'fc2', nh=512, init_scale=np.sqrt(2)))

model = ppo2.learn(env=env, 
                   network=goal_network,
                   nsteps=256,
                   nminibatches=8,
                   lr=5e-4,
                   ent_coef=.01,
                   total_timesteps=int(6e9),
                   log_interval=10,
                   save_interval=100
                  )