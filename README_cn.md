# 介绍  
本项目改造于OpenAI的coinrun项目，原本是一个训练智能体在平台跳跃类型游戏中，躲避障碍和怪物找寻金币的任务。取名courier源于导航任务中的快递员任务，具体来说就是训练智能体能躲避障碍规划路径到达指定位置的任务，类似与快递员运送货物的工作。通过快递员任务实现寻路功能。  

# 主要文件(夹)  
`coinrun/`中包含原始的coinrun项目文件，主要使用了其中的游戏引擎。   
`courier/wrappers.py` 环境包装器，实现快递员任务的核心逻辑。  
`courier/ppo_train.py` 调用`baselines`的接口训练PPO模型。  

# 依赖安装  
```  
sudo apt-get install mpich build-essential qt5-default pkg-config  

pip install numpy gym pyglet mpi4py joblib  
```  
 如需训练，请安装 `baselines`  
```  
pip install git+https://github.com/openai/baselines/archive/7139a66d333b94c2dafc4af35f6a8c7598361df6.zip  
```  

# 运行  
 训练基准的ppo模型  
```  
cd courier  
python ppo_trian.py  
```  
 上手玩一玩  
```  
cd courier  
python play.py  
```  

# 删除怪物  
删除怪物可以减少训练时间，快速验证模型  
通过注释 ```./coinrun/coinrun.cpp``` 中的以下两行实现  
```  
// if (!m->is_walking || (!is_wall(cl) && !is_wall(cr))) // walking monster should have some free space to move  
//   maze->monsters.push_back(m);  
```  