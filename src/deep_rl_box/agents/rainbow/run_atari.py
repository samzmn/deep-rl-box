"""
From the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning"
http://arxiv.org/abs/1710.02298.
"""
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.value import RainbowDqnConvNet
from deep_rl_box.agents.rainbow import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib
from deep_rl_box.utils.schedules import LinearSchedule


environment_name = 'Pong'  # Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.
environment_height = 84  # Environment frame screen height.
environment_width = 84  # Environment frame screen width.
environment_frame_skip = 4  # Number of frames to skip.
environment_frame_stack = 4  # Number of frames to stack.
compress_state = True  # Compress state images when store in experience replay.
replay_capacity = int(1e6)  # Maximum replay size.
min_replay_size = 5000  # Minimum replay size before learning starts.
batch_size = 32  # Sample batch size when updating the neural network.
clip_grad = True  # Clip gradients, default on.
max_grad_norm = 10.0  # Max gradients norm when do gradients clip.
eval_exploration_epsilon = 0.01  # Fixed exploration rate in e-greedy policy for evaluation.

priority_exponent = 0.6  # Priority exponent used in prioritized replay.
importance_sampling_exponent_begin_value = 0.4  # Importance sampling exponent begin value.
importance_sampling_exponent_end_value = 1.0  # Importance sampling exponent end value after decay.
normalize_weights = True  # Normalize sampling weights in prioritized replay.

num_atoms = 51  # Number of elements in the support of the categorical DQN.
v_min = -10.0  # Minimum elements value in the support of the categorical DQN.
v_max = 10.0  # Maximum elements value in the support of the categorical DQN.

n_step = 5  # TD n-step bootstrap.
learning_rate = 0.00025  # Learning rate.
discount = 0.99  # Discount rate.
num_iterations = 100  # Number of iterations to run.
num_train_steps = int(5e5)  # Number of training steps (environment steps or frames) to run per iteration.
num_eval_steps = int(2e4)  # Number of evaluation steps (environment steps or frames) to run per iteration.
max_episode_steps = 108000  # Maximum steps (before frame skip) per episode.
learn_interval = 4  # The frequency (measured in agent steps) to update parameters.
target_net_update_interval = 1000  # The frequency (measured in number of Q network parameter updates) to update target networks.
seed = 42  # Runtime seed.
use_tensorboard = True  # Use Tensorboard to monitor statistics, default on.
actors_on_gpu = True  # Run actors on GPU, default on.
debug_screenshots_interval = 0  # Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.
tag = ''  # Add tag to Tensorboard log file.
results_csv_path = './logs/rainbow_atari_results.csv'  # Path for CSV log file.
checkpoint_dir = './checkpoints'  # Path for checkpoint directory.


def main():
    """Trains Rainbow agent on Atari."""
    print(f'Running Rainbow agent')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    random_state = np.random.RandomState(seed)  # pylint: disable=no-member

    # Create environment.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=environment_name,
            frame_height=environment_height,
            frame_width=environment_width,
            frame_skip=environment_frame_skip,
            frame_stack=environment_frame_stack,
            max_episode_steps=max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=True,
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    print('Environment: ', environment_name)
    print('Action spec: ', train_env.action_space.n)
    print('Observation spec: ', train_env.observation_space.shape)

    state_dim = train_env.observation_space.shape
    action_dim = train_env.action_space.n

    # Test environment and state shape.
    obs, _ = train_env.reset(seed=seed)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (environment_height, environment_width, environment_frame_stack)

    atoms = tf.linspace(v_min, v_max, num_atoms)

    network = RainbowDqnConvNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Test network input and output
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

    if compress_state:

        def encoder(transition):
            return transition._replace(
                s_tm1=replay_lib.compress_array(transition.s_tm1),
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_tm1=replay_lib.uncompress_array(transition.s_tm1),
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    replay = replay_lib.PrioritizedReplay(
        capacity=replay_capacity,
        structure=replay_lib.TransitionStructure,
        priority_exponent=priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=normalize_weights,
        random_state=random_state,
        encoder=encoder,
        decoder=decoder,
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
        environment_name=environment_name, agent_name='Rainbow', save_dir=checkpoint_dir, save_format="tf",
    )
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