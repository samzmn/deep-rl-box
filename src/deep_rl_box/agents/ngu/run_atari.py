"""
From the paper "Never Give Up: Learning Directed Exploration Strategies"
https://arxiv.org/abs/2002.06038.
"""

import queue
import multiprocessing
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import tensorflow as tf



from deep_rl_box.networks.value import NguDqnConvNet, NguNetworkInputs
from deep_rl_box.networks.curiosity import NGURndConvNet, NguEmbeddingConvNet
from deep_rl_box.agents.ngu import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib


environment_name = 'Pong'  # Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest. MontezumaRevenge, Pitfall
environment_height = 84  # Environment frame screen height.
environment_width = 84  # Environment frame screen width.
environment_frame_skip = 4  # Number of frames to skip.
environment_frame_stack = 1  # Number of frames to stack.
compress_state = True  # Compress state images when store in experience replay.
num_actors = 2  # Number of actor processes to run in parallel.
replay_capacity = 20000  # Maximum replay size (in number of unrolls stored). Watch for out of RAM.
min_replay_size = 1000  # Minimum replay size before learning starts (in number of unrolls stored). 6250
clip_grad = True  # Clip gradients, default on.
max_grad_norm = 40.0  # Max gradients norm when do gradients clip.

learning_rate = 0.0001  # Learning rate for adam.
int_learning_rate = 0.0005  # Intrinsic learning rate for adam, this is for embedding and RND predictor networks.
ext_discount = 0.997  # Extrinsic reward discount rate.
int_discount = 0.99  # Intrinsic reward discount rate.
adam_eps = 0.0001  # Epsilon for adam.
unroll_length = 80  # Sequence of transitions to unroll before add to replay.
burn_in = 40  # Sequence of transitions used to pass RNN before actual learning. The effective length of unrolls will be burn_in + unroll_length, two consecutive unrolls will overlap on burn_in steps.
batch_size = 32  # Batch size for learning.

policy_beta = 0.3  # Scalar for the intrinsic reward scale.
num_policies = 32  # Number of directed policies to learn, scaled by intrinsic reward scale beta.

episodic_memory_capacity = 3000  # Maximum size of episodic memory. 30000
reset_episodic_memory = True  # Reset the episodic_memory on every episode, only applicable to actors, default on. From NGU Paper on MontezumaRevenge, Instead of resetting the memory after every episode, we do it after a small number of consecutive episodes, which we call a meta-episode. This structure plays an important role when the agent faces irreversible choices.
num_neighbors = 10  # Number of K-nearest neighbors.
kernel_epsilon = 0.0001  # K-nearest neighbors kernel epsilon.
cluster_distance = 0.008  # K-nearest neighbors cluster distance.
max_similarity = 8.0  # K-nearest neighbors cluster distance.

retrace_lambda = 0.95  # Lambda coefficient for retrace.
transformed_retrace = True  # Transformed retrace loss, default on.

priority_exponent = 0.9  # Priority exponent used in prioritized replay.
importance_sampling_exponent = 0.6  # Importance sampling exponent value.
normalize_weights = True  # Normalize sampling weights in prioritized replay.
priority_eta = 0.9  # Priority eta to mix the max and mean absolute TD errors.

num_iterations = 100  # Number of iterations to run.
num_train_steps = int(5e5)  # Number of training steps (environment steps or frames) to run per iteration, per actor.
num_eval_steps = int(2e4)  # Number of evaluation steps (environment steps or frames) to run per iteration.
max_episode_steps = 108000  # Maximum steps (before frame skip) per episode.
target_net_update_interval = 1500  # The interval (measured in Q network updates) to update target Q networks.
actor_update_interval = 100  # The frequency (measured in actor steps) to update actor local Q network.
eval_exploration_epsilon = 0.01  # Fixed exploration rate in e-greedy policy for evaluation.
seed = 42  # Runtime seed.
use_tensorboard = False  # Use Tensorboard to monitor statistics, default on.
actors_on_gpu = True  # Run actors on GPU, default on.
debug_screenshots_interval = 0  # Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.
tag = ''  # Add tag to Tensorboard log file.
results_csv_path = './logs/ngu_atari_results.csv'  # Path for CSV log file.
checkpoint_dir = './checkpoints'  # Path for checkpoint directory.

assert environment_frame_stack == 1  # Register validator for environment_frame_stack.


def main():
    """Trains NGU agent on Atari."""
    print('Running NGU agent on')
    np.random.seed(seed)
    tf.random.set_seed(seed)

    random_state = np.random.RandomState(seed)  # pylint: disable=no-member

    # Create evaluation environment, like R2D2, we disable terminate-on-life-loss and clip reward.
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
            terminal_on_life_loss=False,
            sticky_action=False,
            clip_reward=False,
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n

    print('Environment: %s', environment_name)
    print('Action spec: %s', action_dim)
    print('Observation spec: %s', state_dim)

    # Test environment and state shape.
    obs, _ = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (environment_height, environment_width, environment_frame_stack)

    # Create network for learner to optimize, actor will use the same network with share memory.
    network = NguDqnConvNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=adam_eps)
    # Create RND target and predictor networks.
    rnd_target_network = NGURndConvNet(state_dim=state_dim, is_target=True)
    rnd_predictor_network = NGURndConvNet(state_dim=state_dim, is_target=False)

    # Create embedding networks.
    embedding_network = NguEmbeddingConvNet(state_dim=state_dim, action_dim=action_dim)

    # Second Adam optimizer for embedding and RND predictor networks.
    intrinsic_optimizer = tf.keras.optimizers.Adam(learning_rate=int_learning_rate, epsilon=adam_eps,)

    # Test network output.
    x = NguNetworkInputs(
        s_t=tf.constant(obs[None, None, ...], dtype=tf.float32),
        a_tm1=tf.zeros([1, 1], dtype=tf.int64),
        ext_r_t=tf.zeros([1, 1], dtype=tf.float32),
        int_r_t=tf.zeros([1, 1], dtype=tf.float32),
        policy_index=tf.zeros([1, 1], dtype=tf.int64),
        hidden_s=network.get_initial_hidden_state(1),
    )
    network_output = network(x)
    assert network_output.q_values.shape == (1, 1, action_dim)
    assert len(network_output.hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent

    if compress_state:

        def encoder(transition):
            return transition._replace(
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    replay = replay_lib.PrioritizedReplay(
        capacity=replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=normalize_weights,
        random_state=random_state,
        time_major=True,
        encoder=encoder,
        decoder=decoder,
    )

    # Create queue to shared transitions between actors and learner
    data_queue = queue.Queue(maxsize=num_actors * 2)

    # Create shared objects so all actor processes can access them
    manager = multiprocessing.Manager()
    # Store copy of latest parameters of the neural network in a shared dictionary, so actors can later access it
    shared_params = manager.dict(
        {
            'network': None,
            'embedding_network': None,
            'rnd_predictor_network': None,
        }
    )

    # Create NGU learner instance
    learner_agent = agent.Learner(
        network=network,
        optimizer=optimizer,
        embedding_network=embedding_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        intrinsic_optimizer=intrinsic_optimizer,
        replay=replay,
        min_replay_size=min_replay_size,
        target_net_update_interval=target_net_update_interval,
        unroll_length=unroll_length,
        burn_in=burn_in,
        retrace_lambda=retrace_lambda,
        transformed_retrace=transformed_retrace,
        priority_eta=priority_eta,
        batch_size=batch_size,
        clip_grad=clip_grad,
        max_grad_norm=max_grad_norm,
        shared_params=shared_params,
    )

    # Create actor environments, actor instances.
    actor_envs = [environment_builder() for _ in range(num_actors)]

    def clone_network(network: tf.keras.Model) -> tf.keras.Model:
        """clones subclassed network's architectures (not weights) from its configuration and returns it"""
        return network.__class__.from_config(network.get_config())

    # Each actor has it's own embedding and RND predictor networks,
    # because we don't want to update these network parameters in the middle of an episode,
    # it will only update these networks at the beginning of an episode.
    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            network=clone_network(network),
            rnd_target_network=clone_network(rnd_target_network),
            rnd_predictor_network=clone_network(rnd_predictor_network),
            embedding_network=clone_network(embedding_network),
            random_state=np.random.RandomState(seed + int(i)),  # pylint: disable=no-member
            ext_discount=ext_discount,
            int_discount=int_discount,
            num_policies=num_policies,
            policy_beta=policy_beta,
            episodic_memory_capacity=episodic_memory_capacity,
            reset_episodic_memory=reset_episodic_memory,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
            num_actors=num_actors,
            action_dim=action_dim,
            unroll_length=unroll_length,
            burn_in=burn_in,
            actor_update_interval=actor_update_interval,
            device=None,
            shared_params=shared_params,
        )
        for i in range(num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.NguEpsilonGreedyActor(
        network=network,
        embedding_network=embedding_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        exploration_epsilon=eval_exploration_epsilon,
        episodic_memory_capacity=episodic_memory_capacity,
        num_neighbors=num_neighbors,
        kernel_epsilon=kernel_epsilon,
        cluster_distance=cluster_distance,
        max_similarity=max_similarity,
        random_state=random_state,
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(environment_name=environment_name, agent_name='NGU', save_dir=checkpoint_dir)
    checkpoint.register_pair(('network', network))
    checkpoint.register_pair(('rnd_target_network', rnd_target_network))
    checkpoint.register_pair(('rnd_predictor_network', rnd_predictor_network))
    checkpoint.register_pair(('embedding_network', embedding_network))

    # Run parallel training N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=num_iterations,
        num_train_steps=num_train_steps,
        num_eval_steps=num_eval_steps,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=results_csv_path,
        use_tensorboard=use_tensorboard,
        tag=tag,
        debug_screenshots_interval=debug_screenshots_interval,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    #freeze_support()
    multiprocessing.set_start_method('spawn')
    main()
