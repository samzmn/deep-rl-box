Deep RL Box
=============================
A collection of Deep Reinforcement Learning algorithms implemented with Tensorflow to solve Atari games and classic control tasks like CartPole, LunarLander, and MountainCar.

The overall project structure was based on DeepMind's [DQN Zoo](https://github.com/deepmind/dqn_zoo). We adapted the code to support Tensorflow, in addition also implemented some SOTA algorithms like PPO, RND, R2D2, and Agent57.


# Content
- [Environment and Requirements](#environment-and-requirements)
- [Implemented Algorithms](#implemented-algorithms)
- [Code Structure](#code-structure)
- [Author's Notes](#authors-notes)
- [Quick Start](#quick-start)
- [Train Agents](#train-agents)
- [Evaluate Agents](#evaluate-agents)
- [Monitoring with Tensorboard](#monitoring-with-tensorboard)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Citing our work](#citing-our-work)


# Environment and Requirements
* Python        3.10.6
* pip           23.0.1
* Tensorflow       2.10.0
* gymnasium    1.1.1
* Tensorboard   2.13.0


# Implemented Algorithms
Step 1: Fundamentals
Actor-Critic (1983) – Understand policy gradient methods and value-based learning.

REINFORCE (1987) – Learn about Monte Carlo policy gradients.

REINFORCE-Baseline (1992) – Grasp variance reduction techniques.

Step 2: Deep Q-Learning
DQN (2013) – Get familiar with deep reinforcement learning and experience replay.

Double-DQN, Prioritized-DQN (2015) – Explore stability improvements in Q-learning.

DRQN (2015) – Learn about recurrent Q-networks for handling partial observability.

Step 3: Advanced Value-Based Methods
Rainbow, C51-DQN, QR-DQN (2017) – Study ensemble methods and distributional RL.

IQN (2018) – Understand implicit quantile networks.

R2D2 (2019), NGU (2019) – Learn memory-efficient Q-learning.

Step 4: Policy Optimization
A2C (2016), PPO (2016) – Study advantage actor-critic and trust region methods.

PPO-ICM, PPO-RND (2018) – Explore intrinsic motivation in RL.

SAC (2018) – Learn soft actor-critic for continuous action spaces.

Step 5: Modern Advances
IMPALA (2021) – Understand scalable distributed RL.

Agent57 (2020) – Dive into the latest breakthroughs in exploration strategies.

## Policy-based RL Algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                                               | Note |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---- |
| `reinforce`          | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `reinforce_baseline` | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `actor_critic`       | [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)          | *    |
| `a2c`                | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) \| [synchronous, deterministic variant of A3C](https://openai.com/blog/baselines-acktr-a2c/)  | P    |
| `sac`                | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning](https://arxiv.org/abs/1801.01290) \| [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207) | P *  |
| `ppo`                | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                                                   | P    |
| `ppo_icm`            | [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)                                | P    |
| `ppo_rnd`            | [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)                                                | P    |
| `impala`             | [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) | P    |
<!-- mdformat on -->


## Value-based RL Algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                               | Note |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `dqn`                | [Human Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)        |      |
| `double_dqn`         | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)                        |      |
| `prioritized_dqn`    | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)                                             |      |
| `drqn`               | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)                   | *    |
| `r2d2`               | [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX) | P    |
| `ngu`                | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)                   | P *  |
| `agent57`            | [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/pdf/2003.13350)                          | P *   |

<!-- mdformat on -->


## Distributional Q Learning Algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                               | Note |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `c51_dqn`            | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)                    |      |
| `rainbow`            | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)            |      |
| `qr_dqn`             | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)            |      |
| `iqn`                | [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)      |      |

<!-- mdformat on -->
**Notes**:
* `P` means support distributed training with multiple actors and a single learner running in parallel (only supports running on a single machine).
* `*` means only tested on Atari Pong or Breakout.

# Code Structure



# Author's Notes
* This project is for education and research purpose only. Where we focus on studying the individual algorithms rather than creating a standard library. If you're looking for a ready to use library for your productive application, this is probably the wrong place.
* Most agents only support episodic environment with discrete action space (except PPO which also supports continuous action space).
* Some code might not be optimal, especially the parts involving Python multiprocessing, as speed of code execution is not our main focus.
* Try our best to replicate the implementation for the original paper, but may change some hyper-parameters to support low budget setup. Also, the hyper-parameters and network architectures are not fine-tuned.
* For Atari games, we only use Pong or Breakout to test the agents, and we stop training once the agent have made some progress.
* We can't guarantee it's bug free. So bug report and pull request are welcome.


# Quick Start

Please check the instructions in the `QUICK_START.md` file on how to setup the project.

# Train Agents

## Classic Control Tasks
* We maintain a list of environment names at `gym_env.py` module, by default it contains ```['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']```.
* For some agents (like advanced DQN agents, most of the policy gradient agents except agents using curiosity-driven exploration), it's impossible to solve MountainCar due to the nature of the problem (sparse reward).

To run a agent on classic control problem, use the following command, replace the <agent_name> with the sub-directory name.
```
python3 -m deep_rl_box.<agent_name>.run_classic

# example of running DQN agents
python3 -m deep_rl_box.dqn.run_classic --environment_name=MountainCar-v0

python3 -m deep_rl_box.dqn.run_classic --environment_name=LunarLander-v2
```

## Atari games
* By default, we uses gym `NoFrameskip-v4` for Atari game, and we omit the need to include 'NoFrameskip' and version in the `environment_name` args, as it will be handled by `create_atari_environment` in the `gym_env.py` module.
* We don't scale the images before store into experience replay, as that will require 2-3x more RAM, we only scale them inside the model.forward() method.

To run a agent on Atari game, use the following command, replace the <agent_name> with the sub-directory name.
```
python3 -m deep_rl_box.<agent_name>.run_atari

# example of running DQN on Atari Pong and Breakout
python3 -m deep_rl_box.dqn.run_atari --environment_name=Pong

python3 -m deep_rl_box.dqn.run_atari --environment_name=Breakout
```

## Distributed training with multiple actors and a single learner (on the same machine)
For agents that support distributed training, we can adjust the parameter `num_actors` to specify how many actors to run.

```
python3 -m deep_rl_box.ppo.run_classic --num_actors=8
```

The following is a high level overview of the distributed training architect. Where each actor has it's own copy of the neural network. And we use the multiprocessing.Queue to transfer the transitions between the actors and the leaner. We also use a shared dictionary to store the latest copy of the neural network's parameters, so the actors can get update it's local copy of the neural network later on.

```
# This will evenly distribute the actors on all GPUs
python3 -m deep_rl_box.ppo.run_atari --num_actors=16 --actors_on_gpu

# This will run all actors on CPU even if you have multiple GPUs
python3 -m deep_rl_box.ppo.run_atari --num_actors=16 --noactors_on_gpu
```


# Evaluate Agents
Before you run the eval_agent module, make sure you have a valid checkpoint file for the specific agent and environment.
By default, it will record a video of agent self-play at the `recordings` directory.

To run a agent on Atari game, use the following command, replace the <agent_name> with the sub-directory name.
```
python3 -m deep_rl_box.<agent_name>.eval_agent

# Example of load pre-trained PPO model on Breakout
python3 -m deep_rl_box.ppo.eval_agent --environment_name=Breakout --load_checkpoint_file=./checkpoints/PPO_Breakout_0.ckpt
```


# Monitoring with Tensorboard
By default, both training, evaluation will log to Tensorboard at the `runs` directory.
To disable this, use the option `--nouse_tensorboard`.

```
tensorboard --logdir=./runs
```

The classes for write logs to Tensorboard is implemented in `trackers.py` module.

* to improve performance, we only write logs at end of episode
* we separate training and evaluation logs
* if algorithm support parallel training, we separate actor, learner logs
* for agents that support parallel training, only log maximum of 8 actors, this is controlled by `run_parallel_training_iterations` in `main_loop.py` module

## Measurements available on Tensorboard
`performance(env_steps)`:
* the statistics are measured over env steps (or frames for Atari), if use frame_skip, it's counted after frame skip
* `episode_return` the non-discounted sum of raw rewards of current episode
* `episode_steps` the current episode length or steps
* `num_episodes` how many episodes have been conducted
* `step_rate(second)` step per seconds, per actors

`agent_statistics(env_steps)`:
* the statistics are measured over env steps (or frames for Atari), if use frame_skip, it's counted after frame skip
* it'll log whatever is exposed in the agent's `statistics` property such as training loss, learning rate, discount, updates etc.
* for algorithm support distributed training (multiple actors), this is only the statistics for the actors.

`learner_statistics(learner_steps)`:
* only available if the agent supports distributed training (multiple actors one learner)
* it'll log whatever is exposed in the learner's `statistics` property such as training loss, learning rate, discount, updates etc.
* to improve performance, it only logs every 100 learner steps

## Add tags to Tensorboard
This could be handy if we want to compare different hyper parameter's performances or different runs with various seeds
```
python3 -m deep_rl_box.impala.run_classic --use_lstm --learning_rate=0.00045 --tag=LSTM-LR0.00045
```

## Debug with environment screenshots
This could be handy if we want to see what's happening during the training, we can set the `debug_screenshots_interval` (measured over number of episode) to some value, and it'll add screenshots of the terminal state to Tensorboard.

```
# Example of creating terminal state screenshot every 100 episodes
python3 -m deep_rl_box.ppo_rnd.run_atari --environment_name=MontezumaRevenge --debug_screenshots_interval=100
```

# Acknowledgments

This project is based on the work of DeepMind, specifically the following projects:
* [DeepMind DQN Zoo](http://github.com/deepmind/dqn_zoo)
* [DeepMind RLax](https://github.com/deepmind/rlax)

In addition, other reference projects from the community have been very helpful to us, including:
* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
* [OpenAI Spinning Up](https://github.com/openai/spinningup)
* [SEED RL](https://github.com/google-research/seed_rl)
* [TorchBeast](https://github.com/facebookresearch/torchbeast)


# License
This project is licensed under the Apache License, Version 2.0 (the "License")
see the LICENSE file for details


# Citing our work
If you reference or use our project in your research, please cite:

```
@software{
  title = {{Deep RL Box}: A collections of Deep RL algorithms implemented with TensorFlow},
  author = {Sam Zamani},
  url = {https://github.com/samzmn/deep_rl_box},
  version = {0.0.1},
  year = {2025},
}
```
