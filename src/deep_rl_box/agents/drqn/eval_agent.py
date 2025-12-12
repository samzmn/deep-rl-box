"""Tests trained DRQN agent from checkpoint with a e-greedy actor.
on classic control tasks like CartPole, MountainCar, or LunarLander, and on Atari."""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.value import DrqnMlpNet, DrqnConvNet
from deep_rl_box.utils import main_loop
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Both Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1. and Atari game like Pong, Breakout.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height, for atari only.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width, for atari only.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip, for atari only.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack, for atari only.')
flags.DEFINE_float('obscure_epsilon', 0.5, 'Make the problem POMDP by obsecure environment state with probability epsilon.')
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('num_iterations', 1, 'Number of evaluation iterations to run.')
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 58000, 'Maximum steps (before frame skip) per episode, for atari only.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('load_checkpoint_file', './checkpoints/drqn/', 'Load a specific checkpoint file.')
flags.DEFINE_string(
    'recording_video_dir',
    'recordings',
    'Path for recording a video of agent self-play.',
)


def main(argv):
    """Tests DRQN agent."""
    del argv
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    random_state = np.random.RandomState(FLAGS.seed)

    # Create evaluation environments
    if FLAGS.environment_name in gym_env.CLASSIC_ENV_NAMES:
        eval_env = gym_env.create_classic_environment(
            env_name=FLAGS.environment_name, obscure_epsilon=FLAGS.obscure_epsilon, seed=random_state.randint(1, 2**10)
        )
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
        network = DrqnMlpNet(state_dim=state_dim, action_dim=action_dim)
    else:
        eval_env = gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            obscure_epsilon=FLAGS.obscure_epsilon,
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
        )
        state_dim = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
        action_dim = eval_env.action_space.n
        network = DrqnConvNet(state_dim=state_dim, action_dim=action_dim)

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Test network input and output. - Call the Model first, then load the weights.
    obs, info = eval_env.reset()
    s = tf.constant(obs[None, None, ...], tf.float32)
    hidden_s = network.get_initial_hidden_state(batch_size=1)
    q = network(s, hidden_s).q_values
    assert q.shape == (1, 1, action_dim)

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = TensorFlowCheckpoint(environment_name=FLAGS.environment_name, agent_name='DRDQN', restore_only=True)
    checkpoint.register_pair(('network', network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)

    network.trainable = False

    # Create evaluation agent instance
    eval_agent = greedy_actors.DrqnEpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
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
