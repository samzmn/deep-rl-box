"""
From the paper "Deep Reinforcement Learning with Double Q-learning"
http://arxiv.org/abs/1509.06461.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

# pylint: disable=import-error
from deep_rl_box.networks.value import DqnConvNet
from deep_rl_box.agents.double_dqn import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib
from deep_rl_box.utils.schedules import LinearSchedule


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_bool('compress_state', True, 'Compress state images when store in experience replay.')
flags.DEFINE_integer('replay_capacity', int(1e6), 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 5000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 32, 'Sample batch size when updating the neural network.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.01, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    int(1e6),
    'Total steps (after frame skip) to decay value of the exploration rate.',
)
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e3), 'Number of training steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e3), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('learn_interval', 4, 'The frequency (measured in agent steps) to update parameters.')
flags.DEFINE_integer(
    'target_net_update_interval',
    500,
    'The frequency (measured in number of Q network parameter updates) to update target networks.',
)
flags.DEFINE_integer('seed', 42, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/double_dqn_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')


def main(argv):
    """Trains Double DQN agent on Atari."""
    del argv
    runtime_device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    logging.info(f'Runs Double DQN agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create environment.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=True,
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', train_env.action_space.n)
    logging.info('Observation spec: %s', train_env.observation_space.shape)

    state_dim = train_env.observation_space.shape
    action_dim = train_env.action_space.n

    # Test environment and state shape.
    obs, _ = train_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == state_dim == (FLAGS.environment_height, FLAGS.environment_width, FLAGS.environment_frame_stack)

    network = DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    # Test network input and output
    q = network(tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)).q_values
    assert q.shape == (1, action_dim)

    # Create e-greedy exploration epsilon schedule
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create transition replay
    if FLAGS.compress_state:

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

    replay = replay_lib.UniformReplay(
        capacity=FLAGS.replay_capacity,
        structure=replay_lib.TransitionStructure,
        random_state=random_state,
        encoder=encoder,
        decoder=decoder,
    )

    # Create DoubleDqn agent instance
    train_agent = agent.DoubleDqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_interval=FLAGS.learn_interval,
        target_net_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discount,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        action_dim=action_dim,
        random_state=random_state,
        device=runtime_device,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        name='Double-Q-greedy',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(environment_name=FLAGS.environment_name, agent_name='Double-Q', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    # Run the training and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        train_agent=train_agent,
        train_env=train_env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    app.run(main)
