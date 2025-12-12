"""
From the paper "A Distributional Perspective on Reinforcement Learning"
http://arxiv.org/abs/1707.06887.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.value import C51DqnMlpNet
from deep_rl_box.agents.c51_dqn import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils.schedules import LinearSchedule
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control tasks name like CartPole-v1, LunarLander-v3, MountainCar-v0, Acrobot-v1.',
)
flags.DEFINE_integer('replay_capacity', 10000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 1000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 64, 'Sample batch size when updating the neural network.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.05, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_decay_step', 10000, 'Total steps to decay value of the exploration rate.')
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')

flags.DEFINE_float('priority_exponent', 0.6, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent_begin_value', 0.4, 'Importance sampling exponent begin value.')
flags.DEFINE_float('importance_sampling_exponent_end_value', 1.0, 'Importance sampling exponent end value after decay.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')

# Parametric Distribution
flags.DEFINE_integer('num_atoms', 51, 'Number of elements in the support of the categorical DQN.')
flags.DEFINE_float('v_min', -10.0, 'Minimum elements value in the support of the categorical DQN.')
flags.DEFINE_float('v_max', 10.0, 'Maximum elements value in the support of the categorical DQN.')

flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 200, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(4e2), 'Number of training env steps to run per iteration.')
flags.DEFINE_integer('num_eval_steps', int(4e2), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('learn_interval', 2, 'The frequency (measured in agent steps) to update parameters.')
flags.DEFINE_integer(
    'target_net_update_interval',
    50,
    'The frequency (measured in number of Q network parameter updates) to update target networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/c51_dqn_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints/c51-dqn', 'Path for checkpoint directory.')


def main(argv):
    """Trains C51-DQN agent on classic control tasks."""
    del argv
    logging.info(f'Runs C51 agent')
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    random_state = np.random.RandomState(FLAGS.seed)

    # Create environment.
    def environment_builder():
        return gym_env.create_classic_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**10),
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', train_env.action_space.n)
    logging.info('Observation spec: %s', train_env.observation_space.shape[0])

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    atoms = tf.cast(tf.linspace(FLAGS.v_min, FLAGS.v_max, FLAGS.num_atoms), dtype=tf.float32)

    network = C51DqnMlpNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=FLAGS.learning_rate, epsilon=1.5e-4)

    # Test network input and output
    obs, info = train_env.reset()
    network_output = network(tf.convert_to_tensor(obs[None, ...], dtype=tf.float32))
    assert network_output.q_logits.shape == (1, action_dim, FLAGS.num_atoms)
    assert network_output.q_values.shape == (1, action_dim)

    # Create e-greedy exploration epsilon schedule
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create prioritized transition replay
    # Note the t in the replay is not exactly aligned with the agent t.
    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        end_t=(FLAGS.num_iterations * int(FLAGS.num_train_steps)),
        begin_value=FLAGS.importance_sampling_exponent_begin_value,
        end_value=FLAGS.importance_sampling_exponent_end_value,
    )
    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=replay_lib.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
    )

    # Create C51Dqn agent instance
    train_agent = agent.C51Dqn(
        network=network,
        optimizer=optimizer,
        atoms=atoms,
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
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        name='C51-DQN-greedy',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='C51-DQN', save_dir=FLAGS.checkpoint_dir
    )
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
        seed=FLAGS.seed,
    )


if __name__ == '__main__':
    app.run(main)
