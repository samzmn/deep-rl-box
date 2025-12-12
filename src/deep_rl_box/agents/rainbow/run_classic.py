"""
From the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning"
http://arxiv.org/abs/1710.02298.
"""
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from deep_rl_box.networks.value import RainbowDqnMlpNet, RainbowSimpleDqnMlpNet, get_rainbow_dqn_mlp_net
from deep_rl_box.agents.rainbow import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib
from deep_rl_box.utils.schedules import LinearSchedule

environment_name = 'LunarLander-v3'  # Classic control tasks name like CartPole-v1, LunarLander-v3, MountainCar-v0, Acrobot-v1.
replay_capacity = 100_000  # Maximum replay size.
min_replay_size = 1000  # Minimum replay size before learning starts.
batch_size = 64  # Sample batch size when updating the neural network.
clip_grad = True  # Clip gradients, default off.
max_grad_norm = 0.5  # Max gradients norm when do gradients clip.
eval_exploration_epsilon = 0.001  # Fixed exploration rate in e-greedy policy for evaluation.

priority_exponent = 0.6  # Priority exponent used in prioritized replay.
importance_sampling_exponent_begin_value = 0.4  # Importance sampling exponent begin value.
importance_sampling_exponent_end_value = 1.0  # Importance sampling exponent end value after decay.
normalize_weights = True  # Normalize sampling weights in prioritized replay.

num_atoms = 51  # Number of elements in the support of the categorical DQN.
v_min = -10.0  # Minimum elements value in the support of the categorical DQN.
v_max = 10.0  # Maximum elements value in the support of the categorical DQN.
sigma_init = 0.1 # initial std dev for Noisy Net

n_step = 3  # TD n-step bootstrap.
learning_rate = 0.00025  # Learning rate.
discount = 0.99  # Discount rate.
num_iterations = 200  # Number of iterations to run.
num_train_steps = int(1e3)  # Number of training env steps to run per iteration.
num_eval_steps = int(1e3)  # Number of evaluation env steps to run per iteration.
learn_interval = 1  # The frequency (measured in agent steps) to update parameters.
target_net_update_interval = 50  # The frequency (measured in number of Q network parameter updates) to update target networks.
seed = 42  # Runtime seed.
use_tensorboard = True  # Use Tensorboard to monitor statistics, default on.
debug_screenshots_interval = 0  # Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.
tag = ''  # Add tag to Tensorboard log file.
results_csv_path = './logs/rainbow_classic_results.csv'  # Path for CSV log file.
checkpoint_dir = './checkpoints/rainbow'  # Path for checkpoint directory.


def main():
    """Trains Rainbow agent on classic control tasks."""
    print(f'Running Rainbow agent')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    random_state = np.random.RandomState(seed)

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

    atoms = tf.cast(tf.linspace(v_min, v_max, num_atoms), dtype=tf.float32)

    # network = RainbowSimpleDqnMlpNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms)
    network = RainbowDqnMlpNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms, units=32, sigma_init=sigma_init)
    # network = get_rainbow_dqn_mlp_net(state_dim=state_dim, action_dim=action_dim, atoms=atoms, units=32, activation="elu", sigma_init=sigma_init)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, epsilon=1.5e-4)
    
    # Test network input and output
    obs, _ = train_env.reset(seed=seed)
    network_output = network(tf.convert_to_tensor(obs[None, ...], dtype=tf.float32))
    assert network_output.q_logits.shape == (1, action_dim, num_atoms)
    assert network_output.q_values.shape == (1, action_dim)

    # Create prioritized transition replay
    # Note the t in the replay is not exactly aligned with the agent t.
    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(min_replay_size),
        end_t=(num_iterations * int(num_train_steps)),
        begin_value=importance_sampling_exponent_begin_value,
        end_value=importance_sampling_exponent_end_value,
    )
    replay = replay_lib.EagerPrioritizedReplay(
        capacity=replay_capacity,
        structure=replay_lib.TransitionStructure,
        priority_exponent=priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=normalize_weights,
        random_state=random_state,
    )

    # Create RainbowDqn agent instance
    train_agent = agent.RainbowDqn(
        network=network,
        optimizer=optimizer,
        atoms=atoms,
        transition_accumulator=replay_lib.NStepTransitionAccumulator(n=n_step, discount=discount),
        replay=replay,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        learn_interval=learn_interval,
        target_net_update_interval=target_net_update_interval,
        n_step=n_step,
        discount=discount,
        clip_grad=clip_grad,
        max_grad_norm=max_grad_norm,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=eval_exploration_epsilon,
        random_state=random_state,
        name='Rainbow-greedy',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(
        environment_name=environment_name, agent_name='Rainbow', save_dir=checkpoint_dir, save_format="keras", min_score=100
    )
    checkpoint.register_pair(('network', network))
    
    # Run the training and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=num_iterations,
        num_train_steps=num_train_steps,
        num_eval_steps=0,
        train_agent=train_agent,
        train_env=train_env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=results_csv_path,
        use_tensorboard=use_tensorboard,
        tag=tag,
        debug_screenshots_interval=debug_screenshots_interval,
        seed=seed,
    )


if __name__ == '__main__':
    main()
