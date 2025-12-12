"""
From the paper "Actor-Critic Algorithms"
https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.
"""
import numpy as np
import tensorflow as tf

# pylint: disable=import-error
from deep_rl_box.networks.policy import ActorCriticMlpNet
from deep_rl_box.agents.actor_critic import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib


# Environment settings
environment_name = "CartPole-v1"  # Classic control tasks (e.g., CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1).

# Training parameters
clip_grad = False  # Clip gradients, default off.
max_grad_norm = 0.5  # Max gradients norm when clipping gradients.
learning_rate = 0.001  # Learning rate.
discount = 0.96  # Discount rate.
entropy_coef = 0.05  # Coefficient for the entropy loss.
value_coef = 0.5  # Coefficient for the state-value loss.
batch_size = 32  # Accumulate batch size transitions before performing learning.

# Iteration and step settings
num_iterations = 500  # Number of iterations to run.
num_train_steps = int(2e2)  # Number of training environment steps per iteration.
num_eval_steps = int(2e2)  # Number of evaluation environment steps per iteration.

# Miscellaneous settings
seed = 42  # Runtime seed.
use_tensorboard = True  # Use TensorBoard to monitor statistics (default on).
actors_on_gpu = True  # Run actors on GPU, default on.
debug_screenshots_interval = 0  # Take screenshots every N episodes and log to TensorBoard (default: 0, no screenshots).
tag = ""  # Add tag to TensorBoard log file.
results_csv_path = "./logs/actor_critic_classic_results.csv"  # Path for CSV log file.
checkpoint_dir = "./checkpoints/actor-critic"  # Path for checkpoint directory.


def main():
    """Trains Actor-Critic agent on classic control tasks."""
    print('Runs Rainbow agent')
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

    # Create policy network and optimizer
    policy_network = ActorCriticMlpNet(state_dim=state_dim, action_dim=action_dim)
    policy_optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    # Test network output.
    obs, _ = train_env.reset(seed=seed)
    s = tf.constant(obs[None, ...], dtype=tf.float32)
    network_output = policy_network(s)
    assert network_output.pi_logits.shape == (1, action_dim)
    assert network_output.value.shape == (1, 1)

    # Create Actor-Critic agent instance
    train_agent = agent.ActorCritic(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        discount=discount,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        clip_grad=clip_grad,
        max_grad_norm=max_grad_norm,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        name='Actor-Critic',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(
        environment_name=environment_name, agent_name='Actor-Critic', save_dir=checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))

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
