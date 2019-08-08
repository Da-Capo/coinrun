# Requirements
```
sudo apt-get install mpich build-essential qt5-default pkg-config

pip install numpy gym pyglet mpi4py joblib
```
 if you want train the ppo model , install the ```baselines```
```
pip install git+https://github.com/openai/baselines/archive/7139a66d333b94c2dafc4af35f6a8c7598361df6.zip
```

# run
 train
```
cd courier
python ppo_trian.py
```
 play and see
```
cd courier
python play.py
```

# remove monster
You can comment this two lines in ```./coinrun/coinrun.cpp``` to remove monster
```
// if (!m->is_walking || (!is_wall(cl) && !is_wall(cr))) // walking monster should have some free space to move
//   maze->monsters.push_back(m);
```