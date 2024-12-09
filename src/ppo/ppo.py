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

lr = 8e-5
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames =  200_000 # 500_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.1  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 4e-4

# register env
register(
    id="MachineEnv-v0",
    entry_point=MachineEnvironment
)

base_env = GymEnv("MachineEnv-v0", device=device)

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
# print("normalization constant shape:", env.transform[0].loc.shape)

# print("observation_spec:", env.observation_spec)
# print("reward_spec:", env.reward_spec)
# print("input_spec:", env.input_spec)
# print("action_spec (as defined by input_spec):", env.action_spec)

check_env_specs(env)

policy_module, value_module = setup_model(env)
policy_module = policy_module.to(device)
value_module = value_module.to(device)

# initialize
policy_module(env.reset())
value_module(env.reset())

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs.flatten()

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )

    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    if i % 5 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            cumulative_reward = eval_rollout["next", "reward"].sum().item()
            eval_slowdown = base_env.mean_slowdown()

            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(cumulative_reward)

            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            logs["eval slowdown"].append(eval_slowdown)

            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )

            del eval_rollout

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

    frame_status = f'{i}/{total_frames / frames_per_batch}'
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str, frame_status]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

plt.ioff()
plt.show()

OUT_DIR = '../out/ppo'
PPO_OUT_DIR = '../out/ppo_model'

os.makedirs(PPO_OUT_DIR, exist_ok=True)

torch.save(policy_module.state_dict(), os.path.join(OUT_DIR, "policy_module.pth"))
torch.save(value_module.state_dict(), os.path.join(OUT_DIR, "value_module.pth"))
torch.save(optim.state_dict(), os.path.join(OUT_DIR, "optimizer.pth"))
torch.save(scheduler.state_dict(), os.path.join(OUT_DIR, "scheduler.pth"))

num_files = 0
for file in os.listdir(OUT_DIR):
    path = os.path.join(OUT_DIR, file)
    if not os.path.isfile(path):
        continue

    num_files += 1

plot_path = os.path.join(OUT_DIR, f'{num_files}.png')
fig.savefig(plot_path)
