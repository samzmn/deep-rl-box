"""Tests trained REINFORCE with value agent from checkpoint with a policy-greedy actor.
on classic control tasks like CartPole, MountainCar, or LunarLander, and on Atari."""
import tensorflow as tf
import numpy as np

from deep_rl_box.networks.policy import ActorMlpNet, ActorConvNet
from deep_rl_box.utils import main_loop
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import gym_env
from deep_rl_box.utils import greedy_actors


# Environment settings
environment_name = "CartPole-v1"  # Both Classic control tasks (e.g., CartPole-v1, LunarLander-v2) and Atari games (e.g., Pong, Breakout).
environment_height = 84  # Environment frame screen height, for Atari only.
environment_width = 84  # Environment frame screen width, for Atari only.
environment_frame_skip = 4  # Number of frames to skip, for Atari only.
environment_frame_stack = 4  # Number of frames to stack, for Atari only.

# Evaluation parameters
num_iterations = 1  # Number of evaluation iterations to run.
num_eval_steps = int(2e4)  # Number of evaluation steps (environment steps or frames) to run per iteration.
max_episode_steps = 58000  # Maximum steps (before frame skip) per episode, for Atari only.

# Miscellaneous settings
seed = 1  # Runtime seed.
use_tensorboard = True  # Use TensorBoard to monitor statistics (default on).
load_checkpoint_file = ""  # Load a specific checkpoint file.
recording_video_dir = "recordings"  # Path for recording a video of agent self-play.



def main():
    """Tests REINFORCE-BASELINE agent."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    random_state = np.random.RandomState(seed)  # pylint: disable=no-member

    # Create evaluation environments
    if environment_name in gym_env.CLASSIC_ENV_NAMES:
        eval_env = gym_env.create_classic_environment(env_name=environment_name, seed=random_state.randint(1, 2**10))
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
        policy_network = ActorMlpNet(state_dim=state_dim, action_dim=action_dim)
    else:
        eval_env = gym_env.create_atari_environment(
            env_name=environment_name,
            frame_height=environment_height,
            frame_width=environment_width,
            frame_skip=environment_frame_skip,
            frame_stack=environment_frame_stack,
            max_episode_steps=max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
        )
        state_dim = (environment_frame_stack, environment_height, environment_width)
        action_dim = eval_env.action_space.n
        policy_network = ActorConvNet(state_dim=state_dim, action_dim=action_dim)

    print('Environment: ', environment_name)
    print('Action spec: ', action_dim)
    print('Observation spec: ', state_dim)

    # Test network output.
    obs, info = eval_env.reset()
    s = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)
    pi_logits = policy_network(s).pi_logits
    value = policy_network(s).value
    assert pi_logits.shape == (1, action_dim)
    assert value.shape == (1, 1)

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = TensorFlowCheckpoint(environment_name=environment_name, agent_name='REINFORCE-BASELINE', restore_only=True)
    checkpoint.register_pair(('policy_network', policy_network))

    if load_checkpoint_file:
        checkpoint.restore(load_checkpoint_file)

    policy_network.trainable = False

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        name='REINFORCE-BASELINE-greedy',
    )

    # Run test N iterations.
    main_loop.run_evaluation_iterations(
        num_iterations=num_iterations,
        num_eval_steps=num_eval_steps,
        eval_agent=eval_agent,
        eval_env=eval_env,
        use_tensorboard=use_tensorboard,
        recording_video_dir=recording_video_dir,
    )


if __name__ == '__main__':
    main()
