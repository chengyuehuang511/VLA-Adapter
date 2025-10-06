import json
import logging
import os
os.environ["MUJOCO_GL"] = "osmesa"
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import multiprocessing
import traceback
import argparse

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

sys.path.append('.')
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from utils.logger import Logger, reset_logging
from utils.visualize import write_video

# from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    num_gpus: int = 8                                # Number of GPUs to use
    num_processes: int = 32                          # Number of parallel processes to use

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    # local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_root: str = "./results"                     # Root directory for saving results

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    fps:  int = 30                                   # Frames per second for saved videos

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"


class ParallelLiberoEvaluator:
    def __init__(self, cfg: GenerateConfig):
        self.cfg = cfg
        
        # Validate configuration
        self._validate_config(cfg)

        # [Note] Tokenizers parallelism is set to true for faster tokenization
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'
        
    def evaluate(self):
        from libero.libero import benchmark

        self._set_results()
        self._build_logger()
        self.logger.infos('Config', vars(self.cfg))

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.cfg.task_suite_name]()
        num_tasks_in_suite = self.task_suite.n_tasks

        gpus = self._check_free_gpus()
        if self.cfg.num_gpus < len(gpus):
            gpus = gpus[:self.cfg.num_gpus]
        
        task_ids_and_episodes_all_processes = [[] for _ in range(self.cfg.num_processes)]
        idx = 0
        for task_id in range(num_tasks_in_suite):
            # task = self.task_suite.get_task(task_id).language
            for episode in range(self.cfg.num_trials_per_task):
                task_ids_and_episodes_all_processes[idx % self.cfg.num_processes].append((task_id, episode))
                idx += 1

        processes = []
        manager = multiprocessing.Manager()
        summaries = manager.list()
        
        for idx, task_ids_and_episodes in enumerate(task_ids_and_episodes_all_processes):
            gpu = gpus[idx % len(gpus)]
            self.logger.info(f'GPU {gpu}: {task_ids_and_episodes}')
            process = multiprocessing.Process(target=self.evaluate_episodes,
                                              args=(gpu, task_ids_and_episodes, idx == 0, summaries))
            processes.append(process)
            
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        self._build_logger(mode='a')
        task_ids = set([summary["task_id"] for summary in summaries])
        for task_id in task_ids:
            task_summaries = [summary for summary in summaries if summary["task_id"] == task_id]
            success_rate = sum([summary["success"] for summary in task_summaries]) / len(task_summaries)
            task_description = task_summaries[0]['task']
            self.logger.info(f"Task {task_id} {task_description} success rate: {success_rate:.2f}")
        
        success_rate = sum([summary["success"] for summary in summaries]) / len(summaries)
        self.logger.info(f"Overall success rate: {success_rate:.2f}")
        self.logger.info("Evaluation finished.")
    
    def evaluate_episodes(self, gpu, task_ids_and_episodes, show_detail, summaries):
        try:
            model, action_head, proprio_projector, noisy_action_projector, processor = self._build_policy(gpu)
            reset_logging()
            self._build_logger(mode='a')

            for task_id, episode in task_ids_and_episodes:
                self.logger.info(f"GPU {gpu}: task {task_id} episode {episode}")
                summary = self.evaluate_single(model, processor, action_head, proprio_projector, noisy_action_projector, task_id, episode, show_detail)
                summaries.append(summary)
        
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            with open(os.path.join(self.save_dir, f'error_gpu{gpu}.log'), 'w') as f:
                f.write(str(e) + '\n')
                f.write(traceback.format_exc())

    def evaluate_single(self, model, processor, action_head, proprio_projector, noisy_action_projector, task_id, episode, show_detail):
        # Get expected image dimensions
        from experiments.robot.robot_utils import get_image_resize_size, get_action
        resize_size = get_image_resize_size(self.cfg)

        task = self.task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, self.cfg.model_family, resolution=self.cfg.env_img_res)
        env.seed(episode)
        env.reset()

        # for libero object, we reset the environment
        # so the initial state is not the same as the training data
        # if not self.cfg.task_suite_name == 'libero_object':
            # initial_states = self.task_suite.get_task_init_states(task_id)
            # obs = env.set_init_state(initial_states[episode])
        
        # Get initial states
        initial_states, all_initial_states = self._load_initial_states(self.cfg, self.task_suite, task_id, self.logger)

        # Handle initial state
        if self.cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                self.logger.error(f"Skipping task {task_id} episode {episode} due to failed expert demo!")
                return {"task_id": task_id, "task": task_description, "episode": episode, "success": False}

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        # Set initial state if provided
        if initial_state is not None:
            obs = env.set_init_state(initial_state)
        else:
            obs = env.get_observation()

        # Initialize action queue
        NUM_ACTIONS_CHUNK = 8  # Default value; replace with actual constant if available
        if self.cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
            print(f"WARNING: cfg.num_open_loop_steps ({self.cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
                "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
                "both speed and success rate), we recommend executing the full action chunk.")
        action_queue = deque(maxlen=self.cfg.num_open_loop_steps)

        # Setup
        t = 0
        replay_images = []
        max_steps = TASK_MAX_STEPS[self.cfg.task_suite_name]

        # Run episode
        success = False
        while t < max_steps + self.cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < self.cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(self.cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = self._prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    self.cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=self.cfg.use_film,
                    use_minivlm=self.cfg.use_minivlm
                )

                action_queue.extend(actions) 

            # Get action from queue
            action = action_queue.popleft()
            # action = actions[0]


            # Process action
            action = self._process_action(action, self.cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if show_detail:
                self.logger.info(f"Step {t}: done {done}, {info}")
            if done:
                success = True
                break
            t += 1
        
        video_save_dir = os.path.join(self.save_dir, f'{task_id}_{task_description}')
        os.makedirs(video_save_dir, exist_ok=True)
        write_video(replay_images, os.path.join(video_save_dir, f'episode{episode}_success={success}.gif'), 
                    texts=None, fps=self.cfg.fps)
        
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
        return {"task_id": task_id, "task": task_description, "episode": episode, "success": success}
            
    def _set_results(self):
        self.save_dir = os.path.join(self.cfg.save_root, 
                                     f'{self.cfg.task_suite_name}-{self.cfg.model_family}')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _build_logger(self, mode='w'):
        self.logger = Logger(os.path.join(self.save_dir, '000.log'), mode=mode)

    def _check_free_gpus(self):
        """ Check free GPUs. """
        used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
        used_memorys = [int(memory.strip()) for memory in used_memorys]
        return [i for i, memory in enumerate(used_memorys) if memory < 1000]

    def _set_gpu(self, gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # list_physical devices can avoid cuda error, don't know why
        import tensorflow as tf
        tf.config.list_physical_devices("GPU")
    
    def _build_policy(self, gpu):
        self._set_gpu(gpu)

        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
        )
        from experiments.robot.robot_utils import get_model, set_seed_everywhere

        set_seed_everywhere(self.cfg.seed)
        
        # Initialize model and components
        # Load model
        model = get_model(self.cfg)
        model.set_version(self.cfg.save_version)
        # Load proprio projector if needed
        proprio_projector = None
        if self.cfg.use_proprio:
            proprio_projector = get_proprio_projector(
                self.cfg,
                model.llm_dim,
                proprio_dim=8,  # 8-dimensional proprio for LIBERO
            )

        # Load action head if needed
        action_head = None
        if self.cfg.use_l1_regression:
            action_head = get_action_head(self.cfg, model.llm_dim)

        # Load noisy action projector if using diffusion
        noisy_action_projector = None

        # Get OpenVLA processor if needed
        processor = None
        if self.cfg.model_family == "openvla":
            processor = get_processor(self.cfg)
            self._check_unnorm_key(self.cfg, model)

        return model, action_head, proprio_projector, noisy_action_projector, processor

    def _validate_config(self, cfg: GenerateConfig) -> None:
        """Validate configuration parameters."""
        assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

        if "image_aug" in str(cfg.pretrained_checkpoint):
            assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

        assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

        # Validate task suite
        assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


    def _check_unnorm_key(self, cfg: GenerateConfig, model) -> None:
        """Check that the model contains the action un-normalization key."""
        # Initialize unnorm_key
        # unnorm_key = cfg.task_suite_name
        unnorm_key = 'libero_90'

        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"

        assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

        # Set the unnorm_key in cfg
        cfg.unnorm_key = unnorm_key


    def _load_initial_states(self, cfg: GenerateConfig, task_suite, task_id: int, logger: Logger):
        """Load initial states for the given task."""
        # Get default initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # If using custom initial states, load them from file
        if cfg.initial_states_path != "DEFAULT":
            with open(cfg.initial_states_path, "r") as f:
                all_initial_states = json.load(f)
            logger.info(f"Using initial states from {cfg.initial_states_path}")
            return initial_states, all_initial_states
        else:
            logger.info("Using default initial states")
            return initial_states, None


    def _prepare_observation(self, obs, resize_size):
        from experiments.robot.openvla_utils import resize_image_for_policy

        """Prepare observation for policy input."""
        # Get preprocessed images
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)

        # Resize images to size expected by model
        img_resized = resize_image_for_policy(img, resize_size)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        # Prepare observations dict
        observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        return observation, img  # Return both processed observation and original image for replay


    def _process_action(self, action, model_family):
        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )

        """Process action before sending to environment."""
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = normalize_gripper_action(action, binarize=True)

        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        if model_family == "openvla":
            action = invert_gripper_action(action)

        return action
    


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    else:
        raise ValueError(f"Cannot convert {v} to boolean.")


def main(args):
    cfg = GenerateConfig(
        model_family=args.model_family,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_trials_per_task=args.num_trials_per_task,
        num_gpus=args.num_gpus,
        num_processes=args.num_processes,
        task_suite_name=args.task_suite_name,
        save_root=args.save_root,
        center_crop=str_to_bool(args.center_crop),
        
        use_l1_regression=args.use_l1_regression,
        use_film=args.use_film,
        use_minivlm=args.use_minivlm,
        use_proprio=args.use_proprio,
        num_images_in_input=args.num_images_in_input,
        use_pro_version=args.use_pro_version
    )
    evaluator = ParallelLiberoEvaluator(cfg)
    evaluator.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--num-processes', type=int, default=32)
    parser.add_argument('--task-suite-name', default='libero_90')
    parser.add_argument('--num-trials-per-task', type=int, default=10)
    parser.add_argument('--pretrained-checkpoint', default='')
    parser.add_argument('--save-root', default='./results')
    parser.add_argument('--center-crop', type=str, default='True')
    parser.add_argument('--use-l1-regression', action='store_true')
    parser.add_argument('--use-proprio', action='store_true')
    parser.add_argument('--use-minivlm', action='store_true')
    parser.add_argument('--use-film', action='store_true')
    parser.add_argument('--num-images-in-input', type=int, default=1)
    parser.add_argument('--model-family', default='openvla')
    parser.add_argument('--use-pro-version', action='store_true')

    args = parser.parse_args()
    main(args)