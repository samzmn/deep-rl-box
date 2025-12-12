"""
From the paper "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.
"""
import numpy as np
import tensorflow as tf

from deep_rl_box.networks.policy import ActorMlpNet
from deep_rl_box.agents.reinforce import agent
from deep_rl_box.utils import replay as replay_lib
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import main_loop
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors


environment_name = "CartPole-v1"  # Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1
normalize_returns = False  # Normalize episode returns, default off
clip_grad = False  # Clip gradients, default off
max_grad_norm = 40.0  # Max gradients norm when doing gradient clipping
learning_rate = 0.0005  # Learning rate
discount = 0.99  # Discount rate
num_iterations = 100  # Number of iterations to run
num_train_steps = int(2e2)  # Number of training environment steps to run per iteration
num_eval_steps = int(2e2)  # Number of evaluation environment steps to run per iteration
seed = 1  # Runtime seed
use_tensorboard = True  # Use Tensorboard to monitor statistics, default on
debug_screenshots_interval = 0  # Take screenshots every N episodes and log to Tensorboard, default 0 (no screenshots)
tag = ""  # Add tag to Tensorboard log file
results_csv_path = None # "./logs/reinforce_classic_results.csv"  # Path for CSV log file
checkpoint_dir = "./REINFORCE/checkpoints"  # Path for checkpoint directory

def main():
    """Trains REINFORCE agent on classic control tasks."""
    print(f'Runs REINFORCE agent')
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

    # Test network output.
    obs, info = train_env.reset()
    pi_logits = policy_network(tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)).pi_logits
    assert pi_logits.shape == (1, action_dim)

    # Create reinforce agent instance
    train_agent = agent.Reinforce(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        discount=discount,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        normalize_returns=normalize_returns,
        clip_grad=clip_grad,
        max_grad_norm=max_grad_norm,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        name='REINFORCE-greedy',
    )

    # Setup checkpoint.
    checkpoint = TensorFlowCheckpoint(
        environment_name=environment_name, agent_name='REINFORCE', save_dir=checkpoint_dir
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
