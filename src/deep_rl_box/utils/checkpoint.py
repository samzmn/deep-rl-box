"""Checkpoint class for Deep RL Box."""
from typing import Mapping, Tuple, Text, Any
import os
import json
from pathlib import Path
import tensorflow as tf

from deep_rl_box.networks.common import *
from deep_rl_box.networks.value import *
from deep_rl_box.networks.curiosity import *
from deep_rl_box.networks.policy import *

class TensorFlowCheckpoint:
    """Simple checkpoint class implementation for TensorFlow.

    Example create checkpoint:

    Create checkpoint instance and register network to the checkpoint internal state

    ```
    checkpoint = TensorFlowCheckpoint(environment_name=FLAGS.environment_name, agent_name='NGU', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    for iteration in range(num_iterations):
        ...
        checkpoint.set_iteration(iteration)
        checkpoint.save()
        ...
    ```


    Example restore checkpoint:

    Create checkpoint instance and register network to the checkpoint internal state

    ```
    checkpoint = TensorFlowCheckpoint(environment_name=FLAGS.environment_name, agent_name='NGU', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    checkpoint.restore(FLAGS.load_checkpoint_file)
    network.eval()

    ```

    """

    def __init__(
        self,
        environment_name: str,
        agent_name: str = 'RLAgent',
        save_dir: str = None,
        iteration: int = 0,
        save_format: str = 'keras',
        min_score: int = 0,
        save_best_only: bool = True,
        restore_only: bool = False,
    ) -> None:
        """
        Args:
            environment_name: the environment name for the running agent.
            agent_name: agent name, default RLAgent.
            save_dir: checkpoint files save directory, default None.
            file_ext: checkpoint file extension.
            iteration: current iteration, default 0.
            save_best_only: defaults to True.
            restore_only: only used for evaluation, will not able to create checkpoints, default off.
            save_format: either 'tf' or 'keras'
        """

        self.save_dir = save_dir
        self.base_path = None
        self.save_format = save_format
        self.min_score = min_score

        if not restore_only and self.save_dir is not None and self.save_dir != '':
            self.base_path = Path(self.save_dir)
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)

        # Stores internal state for checkpoint.
        self.save_best_only = save_best_only
        self.previous_file_paths = None
        self.state = dict()
        self.state["iteration"] = iteration
        self.state["environment_name"] = environment_name
        self.state["agent_name"] = agent_name
        self.state["score"] = min_score

    def register_pair(self, pair: Tuple[Text, Any]) -> None:
        """
        Add a pair of (key, item) to internal state so that later we can save as checkpoint.
        """
        assert isinstance(pair, Tuple)

        key, item = pair
        self.state[key] = item

    def save(self, reward=0) -> str:
        """
        Save TensorFlow model as checkpoint, default file name is {agent_name}_{env_name}_{iteration}.keras, for example A2C_CartPole-v1_0.ckpt

        Returns:
            the full path of checkpoint file.
        """
        if self.base_path is None:
            return
        
        if self.save_best_only:
            if reward < self.state["score"]:
                return
            
        if reward < self.min_score:
            return
        
        self.register_pair(("score", reward))

        file_name = self.get_file_name()
        save_path = self.base_path / file_name

        config = {}
        for key, item in self.state.items():
            if self._is_tf_model(item):
                model_file_name = self.get_file_name(last_part=key)
                if self.save_format == "keras":
                    model_file_name += ".weights.h5"
                model_save_path = self.base_path / model_file_name
                item.save_weights(model_save_path.absolute()) # saves a distinct file beside the config.json
                config[key] = {
                    "model": True,
                    "weights_file_name": model_file_name,
                    "class_name": item.__class__.__name__,
                }
            elif self._is_tf_optimizer(item):
                optimizer_config = item.get_config()
                optimizer_config['class_name'] = item.__class__.__name__
                optimizer_config['optimizer'] = True
                config[key] = optimizer_config
            else:
                config[key] = item

        with open(save_path, 'w') as f:
            json.dump(config, f)

        # Delete previous files saved
        if self.previous_file_paths and self.save_best_only:
            for path in self.previous_file_paths:
                try:
                    os.remove(path) 
                except FileNotFoundError:
                    print(f'file not found. got path: {path}')
        # set current saved files as previous files
        self.previous_file_paths = [save_path.absolute()]
        if self.save_format=="tf":
            self.previous_file_paths.extend(
            [Path(self.base_path, path).absolute() for path in os.listdir(model_save_path.parent) if str(path).startswith(model_file_name)])
        else:
            self.previous_file_paths.append(model_save_path.absolute())
        return save_path

    def restore(self, checkpoit_dir: str) -> None:
        """Try to restore checkpoint from a given checkpoint dirctory."""
        if not checkpoit_dir or os.path.isfile(checkpoit_dir) or not os.path.exists(checkpoit_dir):
            raise ValueError(f'"{checkpoit_dir}" is not a valid checkpoint directory.')

        file_name = self.get_file_name()
        file_name = file_name.replace("0_config.json", "")
        files = [f for f in os.listdir(checkpoit_dir) if f.startswith(file_name) and f.endswith("config.json")]
        if len(files) == 0:
            print(f"'{file_name}' not found in path '{os.path.abspath(checkpoit_dir)}'!!! ")
            return

        files.sort()
        files.sort(key=len)
        file_name = files[-1]
        print(f"restoring {file_name}")
        load_path = os.path.join(checkpoit_dir, file_name)

        with open(load_path, 'r') as f:
            config = json.load(f)
            
        # Needs to match environment_name and agent name
        if config["environment_name"] != self.state["environment_name"]:
            err_msg = f'environment_name "{config["environment_name"]}" and "{self.state["environment_name"]}" mismatch.'
            raise RuntimeError(err_msg)
        if 'agent_name' in config and config["agent_name"] != self.state["agent_name"]:
            err_msg = f'agent_name "{config["agent_name"]}" and "{self.state["agent_name"]}" mismatch.'
            raise RuntimeError(err_msg)

        for key, item in config.items():
            # Only restore state with object key in self.state
            if key not in self.state:
                continue

            if self._is_tf_model(item):
                class_name = item.pop("class_name")
                weights_path = checkpoit_dir + '/' + item.pop("weights_file_name")
                # Using globals() if the class is defined in the global scope
                # if class_name not in globals():
                #     raise ValueError(f"Class {class_name} not found in global scope")
                self.state[key].load_weights(weights_path)

            elif self._is_tf_optimizer(item):
                optimizer_class = getattr(tf.keras.optimizers, item.pop('class_name'))
                self.state[key] = optimizer_class.from_config(item)
            else:
                self.state[key] = item

        self.state["score"] = 0

    def set_iteration(self, iteration) -> None:
        self.state["iteration"] = iteration

    def get_iteration(self) -> int:
        return self.state["iteration"]

    def get_file_name(self, last_part: str = "config.json"):
        return "{}_{}_{}_{}".format(self.state["agent_name"], self.state["environment_name"], self.state["iteration"], last_part)
        
    def _is_tf_model(self, obj) -> bool:
        if isinstance(obj, dict):
            try:
                return obj.pop('model')
            except:
                return False
        return isinstance(obj, tf.keras.Model)
    
    def _is_tf_optimizer(self, obj) -> bool:
        if isinstance(obj, dict):
            try:
                return obj.pop('optimizer')
            except:
                return False
        return isinstance(obj, tf.keras.optimizers.Optimizer)


class AttributeDict(dict):
    """A `dict` that supports getting, setting, deleting keys via attributes."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]
