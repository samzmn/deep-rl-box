"""From the paper "Human Level Control Through Deep Reinforcement Learning"
http://www.nature.com/articles/nature14236.
"""
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.value import DqnMlpNet
from deep_rl_box.agents.dqn import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib
from deep_rl_box.utils.schedules import LinearSchedule


# Environment settings
environment_name = "CartPole-v1"  # Classic control tasks (e.g., CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1).

# Replay buffer settings
replay_capacity = 100000  # Maximum replay size.
min_replay_size = 1000  # Minimum replay size before learning starts.
batch_size = 64  # Sample batch size when updating the neural network.

# Training parameters
clip_grad = False  # Clip gradients, default off.
max_grad_norm = 0.5  # Max gradient norm when clipping gradients.
exploration_epsilon_begin_value = 1.0  # Begin value of the exploration rate in e-greedy policy.
exploration_epsilon_end_value = 0.05  # End (decayed) value of the exploration rate in e-greedy policy.
exploration_epsilon_decay_step = 10000  # Total steps to decay the exploration rate.
eval_exploration_epsilon = 0.01  # Fixed exploration rate for evaluation in e-greedy policy.
learning_rate = 0.0005  # Learning rate.
discount = 0.95  # Discount rate.
num_iterations = 200  # Number of iterations to run.
num_train_steps = int(2e2)  # Number of training environment steps per iteration.
num_eval_steps = int(2e2)  # Number of evaluation environment steps per iteration.
learn_interval = 2  # The frequency (measured in agent steps) to update parameters.
target_net_update_interval = 50  # The frequency (measured in number of Q network parameter updates) to update target networks.

# Miscellaneous settings
seed = 1  # Runtime seed.
use_tensorboard = True  # Use TensorBoard to monitor statistics, default on.
actors_on_gpu = True  # Run actors on GPU, default on.
debug_screenshots_interval = 1  # Take screenshots every N episodes and log to TensorBoard (default: 0, no screenshots).
tag = ""  # Add tag to TensorBoard log file.
results_csv_path = "./logs/dqn_classic_results.csv"  # Path for CSV log file.
checkpoint_dir = "./checkpoints/dqn"  # Path for checkpoint directory.


def main():
    """Trains DQN agent on classic control tasks."""
    print(f'Runs DQN agent')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    random_state = np.random.RandomState(seed)  # pylint: disable=no-member

    # Create environment.
    def environment_builder():
        return gym_env.create_classic_environment(
            env_name=environment_name,
            seed=random_state.randint(1, 2**10),
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    print('Environment: ', environment_name)
    print('Action spec: ', train_env.action_space.n)
    print('Observation spec: ', train_env.observation_space.shape[0])

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    network = DqnMlpNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    # Test network input and output.
    obs, info = train_env.reset()
    q = network(tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)).q_values
    assert q.shape == (1, action_dim)

    # Create e-greedy exploration epsilon schedule.
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(min_replay_size),
        decay_steps=int(exploration_epsilon_decay_step),
        begin_value=exploration_epsilon_begin_value,
        end_value=exploration_epsilon_end_value,
    )

    # Create transition replay.
    replay = replay_lib.UniformReplay(
        capacity=replay_capacity, structure=replay_lib.TransitionStructure, random_state=random_state
    )

    # Create DQN agent instance.
    train_agent = agent.Dqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        learn_interval=learn_interval,
        target_net_update_interval=target_net_update_interval,
        discount=discount,
        clip_grad=clip_grad,
        max_grad_norm=max_grad_norm,
        action_dim=action_dim,
        random_state=random_state,
    )

    # Create evaluation agent instance.
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=eval_exploration_epsilon,
        random_state=random_state,
        name='DQN-greedy',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(environment_name=environment_name, agent_name='DQN', save_dir=checkpoint_dir)
    checkpoint.register_pair(('network', network))

    # Run the training and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=num_iterations,
        num_train_steps=num_train_steps,
        num_eval_steps=num_eval_steps,
        train_agent=train_agent,
        train_env=train_env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=results_csv_path,
        use_tensorboard=use_tensorboard,
        tag=tag,
        debug_screenshots_interval=debug_screenshots_interval,
    )


if __name__ == '__main__':
    main()
