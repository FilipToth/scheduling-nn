from numbers import Number
from typing import Callable
from typing_extensions import Self

import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
import os

from collections import defaultdict

import matplotlib.pyplot as plt

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm


from ppo.models import setup_model
from environments.env import MachineEnvironment

from gymnasium import register

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

run_name = "KPI Rewards, Implicit Timesteps, Always-full Job-queue - Run #3"

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames =  300_000 # 500_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected

OUT_DIR = '../out/ppo'
PPO_OUT_DIR = '../out/ppo_model'

class PPOParams:
    def __init__(self, lr: float, gamma: float, lmbda: float, clip_eps: float, entropy_eps: float, max_grad_norm: float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip_eps = clip_eps
        self.entropy_eps = entropy_eps
        self.max_grad_norm = max_grad_norm

    def default() -> Self:
        return PPOParams(
            lr=6e-5,
            gamma=0.99,
            lmbda=0.95,
            clip_eps=0.1,
            entropy_eps=4e-4,
            max_grad_norm=1.0
        )

class PPOTrainer:
    def __init__(self, params: PPOParams, eval_callback: Callable[[Number, Number, Number, Number], None], step_callback: Callable[[Number, Number, Number, Number], None]):
        self.params = params
        self.eval_callback = eval_callback
        self.step_callback = step_callback
        self.frame = 0

        self._create_env()
        self._setup_modules()

    def _create_env(self):
        # register env
        register(
            id="MachineEnv-v0",
            entry_point=MachineEnvironment
        )

        os.makedirs("../out/", exist_ok=True)
        self.base_env = GymEnv("MachineEnv-v0", device=device, log_path="../out/training.log")

        env = TransformedEnv(
            self.base_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )

        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        check_env_specs(env)
        self.env = env

    def _setup_modules(self):
        policy_module, value_module = setup_model(self.env)
        self.policy_module = policy_module.to(device)
        self.value_module = value_module.to(device)

        # initialize
        self.policy_module(self.env.reset())
        self.value_module(self.env.reset())

        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        self.advantage_module = GAE(
            gamma=self.params.gamma, lmbda=self.params.lmbda, value_network=self.value_module, average_gae=True
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.params.clip_epsilon,
            entropy_bonus=bool(self.params.entropy_eps),
            entropy_coef=self.params.entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.params.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, total_frames // frames_per_batch, 0.0
        )

    def train(self):
        while True:
            reward = self.train_step()
            if reward == None:
                break

    def train_step(self):
        tensordict_data = self.collector.next()
        if tensordict_data == None:
            return None

        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            self.advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data_view.cpu())

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = self.replay_buffer.sample(sub_batch_size)
                loss_vals = self.loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()

                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.params.max_grad_norm)
                self.optim.step()
                self.optim.zero_grad()

        reward = tensordict_data["next", "reward"].mean().item()
        step_count = tensordict_data["step_count"].max().item()
        lr = self.optim.param_groups[0]["lr"]

        if self.frame % 5 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = self.env.rollout(1000, self.policy_module)
                cumulative_reward = eval_rollout["next", "reward"].sum().item()
                eval_slowdown = self.base_env.mean_slowdown()

                eval_reward = eval_rollout["next", "reward"].mean().item()
                eval_step_count = eval_rollout["step_count"].max().item()

                self.eval_callback(eval_reward, cumulative_reward, eval_step_count, eval_slowdown)
                del eval_rollout

        numel = tensordict_data.numel()
        self.step_callback(numel, reward, step_count, lr)

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        self.scheduler.step()
        self.frame += 1

        return reward

    def save(self):
        os.makedirs(PPO_OUT_DIR, exist_ok=True)

        torch.save(self.policy_module.state_dict(), os.path.join(OUT_DIR, "policy_module.pth"))
        torch.save(self.value_module.state_dict(), os.path.join(OUT_DIR, "value_module.pth"))
        torch.save(self.optim.state_dict(), os.path.join(OUT_DIR, "optimizer.pth"))
        torch.save(self.scheduler.state_dict(), os.path.join(OUT_DIR, "scheduler.pth"))


if __name__ == "__main__":
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(run_name)
    axs.flatten()

    pbar = tqdm(total=total_frames)
    eval_str = ""

    num_files = 0
    for file in os.listdir(OUT_DIR):
        path = os.path.join(OUT_DIR, file)
        if not os.path.isfile(path):
            continue

        num_files += 1

    plot_path = os.path.join(OUT_DIR, f'{num_files}.png')
    fig.savefig(plot_path)

    logs = defaultdict(list)

    def eval_callback(eval_reward, cumulative_reward, eval_step_count, eval_slowdown):
        global eval_str

        logs["eval reward"].append(eval_reward)
        logs["eval reward (sum)"].append(cumulative_reward)

        logs["eval step_count"].append(eval_step_count)
        logs["eval slowdown"].append(eval_slowdown)

        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )


    def step_callback(num_el, reward, step_count, lr):
        pbar.update(num_el)
        logs["reward"].append(reward)
        logs["step_count"].append(step_count)
        logs["lr"].append(lr)

        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )

        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        # Update interactive plots
        axs[0, 0].cla()
        axs[0, 0].plot(logs["reward"])
        axs[0, 0].set_title("Mean Training Reward")

        axs[0, 1].cla()
        axs[0, 1].plot(logs["step_count"])
        axs[0, 1].set_title("Max Step Count (Training)")

        axs[1, 0].cla()
        axs[1, 0].plot(logs["eval reward (sum)"])
        axs[1, 0].set_title("Cumulative Reward (Testing)")

        axs[1, 1].cla()
        axs[1, 1].plot(logs["eval slowdown"])
        axs[1, 1].set_title("Mean Job Slowdown (Testing)")

        plt.pause(0.1)

        frame_status = f'{num_el}/{total_frames / frames_per_batch}'
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str, frame_status]))


    params = PPOParams.default()
    trainer = PPOTrainer(params, eval_callback, step_callback)
    trainer.train()

    plt.ioff()
    plt.show()
