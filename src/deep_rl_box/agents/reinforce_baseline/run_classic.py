"""
From the paper "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.
"""
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.policy import ActorMlpNet, CriticMlpNet
from deep_rl_box.agents.reinforce_baseline import agent
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors
from deep_rl_box.utils import replay as replay_lib


# Environment settings
environment_name = "CartPole-v1"  # Environment name (Classic control or Atari games).

# Training settings
normalize_returns = False  # Normalize episode returns, default off.
clip_grad = False  # Clip gradients, default off.
max_grad_norm = 40.0  # Max gradient norm when clipping gradients.
learning_rate = 0.0005  # Learning rate for policy network.
value_learning_rate = 0.0005  # Learning rate for value network.
discount = 0.95  # Discount rate.
num_iterations = 200  # Number of iterations to run.
num_train_steps = int(2e2)  # Number of training environment steps per iteration.
num_eval_steps = int(2e2)  # Number of evaluation environment steps per iteration.

# Runtime settings
seed = 1  # Runtime seed.
use_tensorboard = True  # Use TensorBoard to monitor statistics (default on).
debug_screenshots_interval = 0  # Take screenshots every N episodes and log to TensorBoard (default: 0, no screenshots).
tag = ""  # Add tag to TensorBoard log file.
results_csv_path = None # "./logs/reinforce_baseline_classic_results.csv"  # Path for CSV log file.
checkpoint_dir = "./checkpoints/REINFORCE-baseline"  # Path for checkpoint directory.


def main():
    """Trains REINFORCE-BASELINE agent on classic control tasks."""
    print(f'Runs REINFORCE with value agent')
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
    policy_network = ActorMlpNet(state_dim=state_dim, action_dim=action_dim)
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create value network and optimizer
    value_network = CriticMlpNet(state_dim=state_dim)
    baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=value_learning_rate)

    # Test network output.
    obs, info = train_env.reset()
    s = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)
    pi_logits = policy_network(s).pi_logits
    value = value_network(s).value
    assert pi_logits.shape == (1, action_dim)
    assert value.shape == (1, 1)

    # Create reinforce with value agent instance
    train_agent = agent.ReinforceBaseline(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        discount=discount,
        value_network=value_network,
        baseline_optimizer=baseline_optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        normalize_returns=normalize_returns,
        clip_grad=clip_grad,
        max_grad_norm=max_grad_norm,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        name='REINFORCE-BASELINE-greedy',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(
        environment_name=environment_name, agent_name='REINFORCE-BASELINE', save_dir=checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))
    checkpoint.register_pair(('value_network', value_network))

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
