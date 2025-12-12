"""Tests trained Rainbow agent from checkpoint with a e-greedy actor.
on classic control tasks like CartPole, MountainCar, or LunarLander, and on Atari."""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.value import RainbowDqnMlpNet, RainbowDqnConvNet, get_rainbow_dqn_mlp_net
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'LunarLander-v3',
    'Both Classic control tasks name like CartPole-v1, LunarLander-v3, MountainCar-v0, Acrobot-v1. and Atari game like Pong, Breakout.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height, for atari only.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width, for atari only.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip, for atari only.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack, for atari only.')
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('num_atoms', 51, 'Number of elements in the support of the categorical DQN.')
flags.DEFINE_float('v_min', -10.0, 'Minimum elements value in the support of the categorical DQN.')
flags.DEFINE_float('v_max', 10.0, 'Maximum elements value in the support of the categorical DQN.')
flags.DEFINE_integer('num_iterations', 10, 'Number of evaluation iterations to run.')
flags.DEFINE_integer(
    'num_eval_steps', int(1e3), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 58000, 'Maximum steps (before frame skip) per episode, for atari only.')
flags.DEFINE_integer('seed', None, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', False, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('load_checkpoint_file', './checkpoints/rainbow', 'Load a specific checkpoint file.')
flags.DEFINE_string(
    'recording_video_dir',
    'recordings',
    'Path for recording a video of agent self-play.',
)


def main(argv):
    """Tests Rainbow agent."""
    del argv
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    random_state = np.random.RandomState(FLAGS.seed)

    # Create evaluation environments
    atoms = tf.linspace(FLAGS.v_min, FLAGS.v_max, FLAGS.num_atoms)
    if FLAGS.environment_name in gym_env.CLASSIC_ENV_NAMES:
        eval_env = gym_env.create_classic_environment(env_name=FLAGS.environment_name, seed=random_state.randint(1, 2**10))
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
        network = get_rainbow_dqn_mlp_net(state_dim=state_dim, action_dim=action_dim, atoms=atoms, units=32, activation="elu")
    else:
        eval_env = gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
        )
        state_dim = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
        action_dim = eval_env.action_space.n
        network = RainbowDqnConvNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms)

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = TensorFlowCheckpoint(environment_name=FLAGS.environment_name, agent_name='Rainbow', restore_only=True)
    checkpoint.register_pair(('network', network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)

    network.trainable = False

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        name='Rainbow-greedy',
    )

    # Run test N iterations.
    main_loop.run_evaluation_iterations(
        num_iterations=FLAGS.num_iterations,
        num_eval_steps=FLAGS.num_eval_steps,
        eval_agent=eval_agent,
        eval_env=eval_env,
        use_tensorboard=FLAGS.use_tensorboard,
        recording_video_dir=FLAGS.recording_video_dir,
    )


if __name__ == '__main__':
    app.run(main)
