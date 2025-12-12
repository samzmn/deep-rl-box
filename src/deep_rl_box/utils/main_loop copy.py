"""Training loops for Deep RL Zoo."""
from typing import Iterable, List, Tuple, Text, Mapping, Any
import itertools
import collections
import sys
import time
import signal
import queue
import math
import multiprocessing as mp
import threading
from absl import logging
import gym
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=import-error
import deep_rl_box.utils.trackers as trackers_lib
import deep_rl_box.utils.types as types_lib
from deep_rl_box.utils.log import CsvWriter
from deep_rl_box.utils.checkpoint import TensorFlowCheckpoint
from deep_rl_box.utils import gym_env

def run_env_loop(
    agent: types_lib.Agent, env: gym.Env, max_steps: int
) -> List[Tuple[gym.Env, types_lib.TimeStep, types_lib.Agent, types_lib.Action]]:
    """Repeatedly alternates step calls on environment and agent.

    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode.

    Args:
      agent: Agent to be run, has methods `step(timestep)` and `reset()`.
      env: Environment to run, has methods `step(action)` and `reset()`.

    Yields:
      Tuple `(env, timestep_t, agent, a_t)` where
        `a_t = agent.step(timestep_t)`.

    Raises:
        RuntimeError if the `agent` is not an instance of types_lib.Agent.
    """

    if not isinstance(agent, types_lib.Agent):
        raise RuntimeError(f'Expect agent to be an instance of types_lib.Agent. got: {type(agent)}')
    
    statistic_sequence = []

    # For each episode.
    agent.reset()
    # Think of reset as a special 'action' the agent takes, thus given us a reward 'zero', and a new state 's_t'.
    observation, info = env.reset()
    reward = 0.0
    done = loss_life = False
    first_step = True
    info = {}

    for step in range(max_steps):  # For each step in the current episode.
        timestep_t = types_lib.TimeStep(
            observation=observation,
            reward=reward,
            done=done or loss_life,
            first=first_step,
            info=info,
        )
        a_t = agent.step(timestep_t)
        statistic_sequence.append((env, timestep_t, agent, a_t))

        a_tm1 = a_t
        observation, reward, done, truncated, info = env.step(np.array(a_tm1).item())

        first_step = False

        # For Atari games, check if should treat loss a life as a soft-terminal state
        loss_life = False
        if 'loss_life' in info and info['loss_life']:
            loss_life = info['loss_life']

        if done or truncated or loss_life:  # Actual end of an episode
            # This final agent.step() will ensure the done state and final reward will be seen by the agent and the trackers
            timestep_t = types_lib.TimeStep(
                observation=observation,
                reward=reward,
                done=True,
                first=False,
                info=info,
            )
            unused_a = agent.step(timestep_t)  # noqa: F841
            statistic_sequence.append((env, timestep_t, agent, a_t))
            break
    return statistic_sequence, step + 1


def run_env_steps(num_steps: int, agent: types_lib.Agent, env: gym.Env, trackers: Iterable[Any]) -> Mapping[Text, float]:
    """Run some steps and return the statistics, this could be either training, evaluation, or testing steps.

    Args:
        max_episode_steps: maximum steps per episode.
        agent: agent to run, expect the agent to have step(), reset(), and a agent_name property.
        train_env: training environment.
        trackers: statistics trackers.

    Returns:
        A Dict contains statistics about the result.

    """
    seq, last_step = run_env_loop(agent, env, num_steps)
    stats = trackers_lib.generate_statistics(trackers, seq)
    return stats, last_step


def run_single_thread_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    train_agent: types_lib.Agent,
    train_env: gym.Env,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    checkpoint: TensorFlowCheckpoint,
    csv_file: str,
    use_tensorboard: bool,
    tag: str = None,
    debug_screenshots_interval: int = 0,
) -> None:
    """Runs single-thread training and evaluation for N iterations.
    The same code structure is shared by most single-threaded DQN agents,
    and some policy gradients agents like reinforce, actor-critic.

    For every iteration:
        1. Start to run agent for num_train_steps training environment steps/frames.
        2. Create checkpoint file.
        3. (Optional) Run some evaluation steps with a separate evaluation actor and environment.

    Args:
        num_iterations: number of episodes to run.
        num_train_steps: number of frames (or env steps) to run, per iteration.
        num_eval_steps: number of evaluation frames (or env steps) to run, per iteration.
        train_agent: training agent, expect the agent to have step(), reset(), and a agent_name property.
        train_env: training environment.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        checkpoint: checkpoint object.
        csv_file: csv log file path and name.
        use_tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag, default None.
        debug_screenshots_interval: the frequency to take screenshots and add to tensorboard, default 0 no screenshots.

    """
    
    # Create log file writer.
    #writer = CsvWriter(csv_file)

    # Create trackers for training and evaluation
    train_tb_log_prefix = (
        get_tb_log_prefix(train_env.spec.id, train_agent.agent_name, tag, 'train') if use_tensorboard else None
    )
    train_trackers = trackers_lib.make_default_trackers(train_tb_log_prefix, debug_screenshots_interval)

    should_run_evaluator = False
    eval_trackers = None
    if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
        should_run_evaluator = True
        eval_tb_log_prefix = (
            get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, tag, 'eval') if use_tensorboard else None
        )
        eval_trackers = trackers_lib.make_default_trackers(eval_tb_log_prefix, debug_screenshots_interval)

    #for debugging purpose onle - Keep Track of Best Score 
    best_train_score = -np.inf
    best_eval_score = 0
    scores = []

    # Start training
    for iteration in range(1, num_iterations + 1):
        #logging.info(f'Training iteration {iteration}')

        # Run training steps.
        train_stats, train_last_step = run_env_steps(num_train_steps, train_agent, train_env, train_trackers)

        score = np.array(train_stats['mean_episode_reward']).item()
        checkpoint.set_iteration(iteration)
        saved_ckpt = checkpoint.save(score)
        scores.append(score)
        if score >= best_train_score:
            best_train_score = score

        if saved_ckpt:
            #logging.info(f'\n\nNew checkpoint created at "{saved_ckpt}"\n')
            pass

        try:
            exploration_epsilon = train_agent.exploration_epsilon
        except:
            exploration_epsilon = 0.0

        # Logging training statistics.
        log_output = [
            f'iteration: {iteration:3d}',
            f'best_train_score: {best_train_score:3.2f}',
            #f'train_max_step: {num_train_steps:4d}',
            #f'train_last_step: {train_last_step:4d}',
            'train_episode_reward: {:3.2f}'.format(np.array(train_stats['mean_episode_reward']).item()),
            #'train_num_episodes: {:3d}'.format(train_stats['num_episodes']),
            #'train_step_rate: {:4.0f}'.format(train_stats['step_rate']),
            #'train_duration: {:.2f}'.format(train_stats['duration']),
            f'eps: {exploration_epsilon:1.3f}'
        ]

        # Run evaluation steps.
        if should_run_evaluator is True:
            #logging.info(f'Evaluation iteration {iteration}')

            # Run some evaluation steps.
            eval_stats, eval_last_step = run_env_steps(num_eval_steps, eval_agent, eval_env, eval_trackers)

            # Logging evaluation statistics.
            eval_output = [
                #f'eval_max_step: {num_eval_steps:5d}',
                #f'eval_last_step: {eval_last_step:5d}',
                'eval_episode_reward: {:2.2f}'.format(np.array(eval_stats['mean_episode_reward']).item()),
                #'train_num_episodes: {:3d}'.format(eval_stats['num_episodes']),
                #'train_step_rate: {:4.0f}'.format(eval_stats['step_rate']),
                #'train_duration: {:.2f}'.format(eval_stats['duration']),
            ]
            log_output.extend(eval_output)

        log_output_str = ' - '.join(log_output)
        # logging.info(log_output_str)
        print(f"\r{log_output_str}", end="")
        #writer.write(collections.OrderedDict(log_output))
    #writer.close()
    # extra code – this cell plots the learning curve
    plt.figure(figsize=(8, 4))
    plt.plot(scores)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.grid(True)
    plt.show()



def run_parallel_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    learner_agent: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    actors: List[types_lib.Agent],
    actor_envs: List[gym.Env],
    data_queue: mp.Queue,
    checkpoint: TensorFlowCheckpoint,
    csv_file: str,
    use_tensorboard: bool,
    tag: str = None,
    debug_screenshots_interval: int = 0,
) -> None:
    """This is the place to kick start parallel training with multiple actors processes and a single learner process.
    The actual flow is controlled by `run_learner`.

    Args:
        num_iterations: number of iterations to run.
        num_train_steps: number of frames (or env steps) to run, per iteration.
        num_eval_steps: number of evaluation frames (or env steps) to run, per iteration.
        learner_agent: learner agent, expect the agent to have run_train_loop() method.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        actors: list of actor instances we wish to run.
        actor_envs: list of gym.Env for each actor to run.
        data_queue: a multiprocessing.Queue used to receive transition samples from actors.
        checkpoint: checkpoint object.
        csv_file: csv log file path and name.
        use_tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag, default None.
        debug_screenshots_interval: the frequency to take screenshots and add to tensorboard, default 0 no screenshots.

    """
    # Initialize logging.
    init_absl_logging()
    
    # Listen to signals to exit process.
    handle_exit_signal()

    # Create shared iteration count and start, end training event.
    # start_iteration_event is used to signaling actors to run one training iteration,
    # stop_event is used to signaling actors the end of training session.
    # The start_iteration_event and stop_event are only set by the main process.
    iteration_count = mp.Value('i', 0) # signed integer
    start_iteration_event = mp.Event()
    stop_event = mp.Event()

    # To get training statistics from each actor and the learner. We use a single writer to write to csv file.
    log_queue = mp.SimpleQueue()

    # Run learner train loop on a new thread.
    learner = threading.Thread(
        target=run_learner,
        args=(
            num_iterations,
            num_eval_steps,
            learner_agent,
            eval_agent,
            eval_env,
            data_queue,
            log_queue,
            iteration_count,
            start_iteration_event,
            stop_event,
            checkpoint,
            len(actors),
            use_tensorboard,
            tag,
        ),
        daemon=True
    )
    learner.start()

    # Start logging on a new thread, since it's very light-weight task.
    logger = mp.Process(
        target=run_logger,
        args=(log_queue, csv_file),
        daemon=True
    )
    logger.start()

    # Create and start actor processes once, this will preserve actor's internal state like steps etc.
    # Tensorboard log dir prefix.
    num_actors = len(actors)
    actor_tb_log_prefixes = [None for _ in range(num_actors)]
    if use_tensorboard:
        # To get better performance, we only log a maximum of 8 actor statistics to tensorboard
        _step = 1 if num_actors <= 8 else math.ceil(num_actors / 8)
        for i in range(0, num_actors, _step):
            actor_tb_log_prefixes[i] = get_tb_log_prefix(actor_envs[i].spec.id, actors[i].agent_name, tag, 'train')

    processes = []
    for actor, actor_env, tb_log_prefix in zip(actors, actor_envs, actor_tb_log_prefixes):
        p = threading.Thread(
            target=run_actor,
            args=(
                actor,
                actor_env,
                data_queue,
                log_queue,
                num_train_steps,
                iteration_count,
                start_iteration_event,
                stop_event,
                tb_log_prefix,
                debug_screenshots_interval,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all actor to be finished.
    for p in processes:
        p.join()
        #p.close()

    # learner.join()
    logger.join()

    # Close queue.
    data_queue.close()


def run_actor(
    actor: types_lib.Agent,
    actor_env: gym.Env,
    data_queue: mp.Queue,
    log_queue: mp.SimpleQueue,
    num_train_steps: int,
    iteration_count: mp.Value, # type: ignore
    start_iteration_event: mp.Event, # type: ignore
    stop_event: mp.Event, # type: ignore
    tb_log_prefix: str = None,
    debug_screenshots_interval: int = 0,
) -> None:
    """
    Run actor process for as long as required, only terminate if the `stop_event` is set to True.
    Which is set by the main process.

    * Each actor will wait for the `start_iteration_event` signal to start run num_train_steps steps (for one iteration).
    * The actor whoever finished the current iteration first will reset `start_iteration_event` to False,
    so it does not run into a loop that is out of control.

    Args:
        actor: the actor to run.
        actor_env: environment for the actor instance.
        data_queue: multiprocessing.Queue used for transferring data from actor to learner.
        log_queue: multiprocessing.SimpleQueue used for transferring training statistics from actor,
            this is only for write to csv file, not for tensorboard.
        num_train_steps: number of frames (or env steps) to run for one iteration.
        iteration: a counter which is updated by the main process.
        start_iteration_event: start training signal, set by the main process, clear by actor.
        stop_event: end training signal.
        tb_log_prefix: tensorboard run log dir prefix.
        debug_screenshots_interval: the frequency to take screenshots and add to tensorboard, default 0 no screenshots.

    Raises:
        RuntimeError if the `actor` is not a instance of types_lib.Agent.
    """
    
    if not isinstance(actor, types_lib.Agent):
        raise RuntimeError('Expect actor to be a instance of types_lib.Agent.')
    
    actor_trackers = trackers_lib.make_default_trackers(tb_log_prefix, debug_screenshots_interval)

    while not stop_event.is_set():
        # Wait for start training event signal, which is set by the main process.
        if not start_iteration_event.is_set():
            continue

        logging.info(f'Starting {actor.agent_name} ...')
        iteration = iteration_count.value

        # Run training steps.
        train_stats, last_step = run_env_steps(num_train_steps, actor, actor_env, actor_trackers)

        # Mark work done to avoid infinite loop in `run_learner_loop`,
        # also possible multiprocessing.Queue deadlock.
        data_queue.put('PROCESS_DONE', timeout=1)

        # Whoever finished one iteration first will clear the start training event.
        if start_iteration_event.is_set():
            start_iteration_event.clear()

        # Logging statistics after training finished
        log_output = [
            ('iteration', iteration, '%3d'),
            ('role', actor.agent_name, '%2s'),
            #('step', num_train_steps, '%5d'),
            #('last_step', last_step, '%5d'),
            ('episode_reward', train_stats['mean_episode_reward'], '% 2.2f'),
            #('num_episodes', train_stats['num_episodes'], '%3d'),
            #('step_rate', train_stats['step_rate'], '%4.0f'),
            #('duration', train_stats['duration'], '%.2f'),
        ]

        # Add training statistics to log queue, so the logger process can write to csv file.
        log_queue.put(log_output)

def run_learner(
    num_iterations: int,
    num_eval_steps: int,
    learner: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    data_queue: mp.Queue,
    log_queue: mp.SimpleQueue,
    iteration_count: mp.Value, # type: ignore
    start_iteration_event: mp.Event, # type: ignore
    stop_event: mp.Event, # type: ignore
    checkpoint: TensorFlowCheckpoint,
    num_actors: int,
    use_tensorboard: bool,
    tag: str = None,
) -> None:
    """Run learner for N iterations.

    For every iteration:
        1. Signal actors to start a new iteration.
        2. Start to run the learner loop until all actors are finished their work.
        3. Create checkpoint file.
        4. (Optional) Run evaluation steps with a separate evaluation actor and environment.

    At the beginning of every iteration, learner will set the `start_iteration_event` to True, to signal actors to start training.
    The actor whoever finished the iteration first will reset `start_iteration_event` to False.
    Then on the next iteration, the learner will set the `start_iteration_event` to True.

    Args:
        num_iterations: number of iterations to run.
        num_eval_steps: number of evaluation frames (or env steps) to run, per iteration.
        learner: learner agent, expect the agent to have run_train_loop() method.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        data_queue: a multiprocessing.Queue used receive samples from actor.
        log_queue: a multiprocessing.SimpleQueue used send evaluation statistics to logger.
        start_iteration_event: a multiprocessing.Event signal to actors for start training.
        checkpoint: checkpoint object.
        num_actors: number of actors running, used to check if one iteration is over.
        use_tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag.

    Raises:
        RuntimeError if the `learner` is not a instance of types_lib.Learner.
    """
    if not isinstance(learner, types_lib.Learner):
        raise RuntimeError('Expect learner to be a instance of types_lib.Learner.')

    # Create trackers for learner and evaluator
    learner_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, learner.agent_name, tag, 'train') if use_tensorboard else None
    learner_trackers = trackers_lib.make_learner_trackers(learner_tb_log_prefix)
    for tracker in learner_trackers:
        tracker.reset()

    should_run_evaluator = False
    eval_trackers = None
    if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
        should_run_evaluator = True
        eval_tb_log_prefix = (
            get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, tag, 'eval') if use_tensorboard else None
        )
        eval_trackers = trackers_lib.make_default_trackers(eval_tb_log_prefix)

    # Start training
    for iteration in range(1, num_iterations + 1):
        logging.info(f'Training iteration {iteration}')
        logging.info(f'Starting {learner.agent_name} ...')

        # Update shared iteration count.
        iteration_count.value = iteration

        # Set start training event.
        start_iteration_event.set()
        learner.reset()

        run_learner_loop(learner, data_queue, num_actors, learner_trackers)
        
        start_iteration_event.clear()
        # checkpoint.set_iteration(iteration)
        # saved_ckpt = checkpoint.save()

        # Run evaluation steps.
        if should_run_evaluator is True:
            logging.info(f'Evaluation iteration {iteration}')

            # Run some evaluation steps.
            eval_stats, last_step = run_env_steps(num_eval_steps, eval_agent, eval_env, eval_trackers)

            # Logging evaluation statistics
            log_output = [
                #('iteration', iteration, '%3d'),
                ('role', 'evaluation', '%3s'),
                #('step', num_eval_steps, '%5d'),
                #('last_step', last_step, '%5d'),
                ('episode_reward', eval_stats['mean_episode_reward'], '%2.2f'),
                #('num_episodes', eval_stats['num_episodes'], '%3d'),
                #('step_rate', eval_stats['step_rate'], '%4.0f'),
                #('duration', eval_stats['duration'], '%.2f'),
            ]
            log_queue.put(log_output)
            print(f"Evaluation Episode Reward: {eval_stats['mean_episode_reward']}")
            checkpoint.set_iteration(iteration)
            saved_ckpt = checkpoint.save(eval_stats['mean_episode_reward'])

        time.sleep(5)

    # Signal actors training session ended.
    stop_event.set()
    # Signal logger training session ended, using stop_event seems not working.
    log_queue.put('PROCESS_DONE')


def run_learner_loop(
    learner: types_lib.Learner,
    data_queue: mp.Queue,
    num_actors: int,
    learner_trackers: Iterable[Any],
) -> None:
    """
    Run learner loop by constantly pull item off multiprocessing.queue and calls the learner.step() method.
    """

    num_done_actors = 0

    # Run training steps.
    while True:
        # Try to pull one item off multiprocessing.queue.
        try:
            item = data_queue.get(timeout=1)
            if item == 'PROCESS_DONE':  # one actor process is done for current iteration
                num_done_actors += 1
            else:
                learner.received_item_from_queue(item)
        except queue.Empty:
            pass
        except EOFError:
            pass
        
        # Only break if all actor processes are done
        if num_done_actors == num_actors:
            break

        # The returned stats_sequences could be None when call learner.step(), since it will perform internal checks.
        stats_sequences = learner.step()
        
        if stats_sequences is not None:
            # Some agents may perform multiple network updates in a single call to method step(), like PPO.
            for stats in stats_sequences:
                for tracker in learner_trackers:
                    tracker.step(stats)


def run_logger(log_queue: mp.SimpleQueue, csv_file: str):
    """Run logger and csv file writer on a separate thread,
    this is only for training/evaluation statistics."""

    # Create log file writer.
    writer = CsvWriter(csv_file)

    while True:
        try:
            log_output = log_queue.get()
            if log_output == 'PROCESS_DONE':
                break
            log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
            logging.info(log_output_str)
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
        except queue.Empty:
            pass
        except EOFError:
            pass


def run_evaluation_iterations(
    num_iterations: int,
    num_eval_steps: int,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    use_tensorboard: bool,
    recording_video_dir: str = None,
):
    """Testing an agent restored from checkpoint.

    Args:
        num_iterations: number of iterations to run.
        num_eval_steps: number of evaluation steps, per iteration.
        eval_agent: evaluation agent, expect the agent has step(), reset(), and agent_name property.
        eval_env: evaluation environment.
        use_tensorboard: if True, use tensorboard to log the runs.
        recording_video_dir: folder to store agent self-play video for one episode.
    """

    # Tensorboard log dir prefix.
    test_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, None, 'test') if use_tensorboard else None
    test_trackers = trackers_lib.make_default_trackers(test_tb_log_prefix)

    if num_iterations > 0 and num_eval_steps > 0:
        for iteration in range(1, num_iterations + 1):
            logging.info(f'Testing iteration {iteration}')

            # Run some testing steps.
            eval_stats, last_step = run_env_steps(num_eval_steps, eval_agent, eval_env, test_trackers)

            # Logging testing statistics.
            log_output = [
                ('iteration', iteration, '%3d'),
                ('step', num_eval_steps, '%5d'),
                ('last_step', last_step, '%5d'),
                ('episode_reward', eval_stats['mean_episode_reward'], '% 2.2f'),
                ('num_episodes', eval_stats['num_episodes'], '%3d'),
                ('step_rate', eval_stats['step_rate'], '%4.0f'),
                ('duration', eval_stats['duration'], '%.2f'),
            ]

            log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
            logging.info(log_output_str)
            iteration += 1

    if recording_video_dir is not None and recording_video_dir != '':
        gym_env.play_and_record_video(eval_agent, eval_env, recording_video_dir)


def get_tb_log_prefix(env_id: str, agent_name: str, tag: str, suffix: str) -> str:
    """Returns the composed tensorboard log prefix,
    which is in the format {env_id}-{agent_name}-{tag}-{suffix}."""
    tb_log_prefix = f'{env_id}-{agent_name}'
    if tag is not None and tag != '':
        tb_log_prefix += f'-{tag}'
    tb_log_prefix += f'-{suffix}'
    return tb_log_prefix


def init_absl_logging():
    """Initialize absl.logging when run the process without app.run()"""
    logging._warn_preinit_stderr = 0  # pylint: disable=protected-access
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    # signal.signal(signal.SIGHUP, shutdown) # Hangup detected on controlling terminal or death of controlling process. available on Unix
    signal.signal(signal.SIGINT, shutdown) # Interrupt from keyboard (CTRL + C).
    signal.signal(signal.SIGTERM, shutdown) # Termination signal.
