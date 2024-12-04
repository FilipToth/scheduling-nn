import torch
from gymnasium import register
from torch import multiprocessing
from ppo.models import setup_model
from torchrl.envs.libs.gym import GymEnv
from environments.env import MachineEnvironment
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

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

# I have to load the .pth files...

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
env.set_seed(5051)

policy_module, _ = setup_model(env)
policy_module = policy_module.to(device)

policy_net_state = torch.load("../out/ppo/policy_module.pth")
policy_module.load_state_dict(policy_net_state)

policy_module(env.reset())
policy_module.eval()

with torch.no_grad():
    print("EVALUATING MODEL...")

    initial = env.reset()
    print(initial["observation"])
    rollout = env.rollout(max_steps=1000, policy=policy_module, tensordict=initial, auto_reset=False)

    slowdown = base_env.mean_slowdown()
    print(f"mean slowdown: {slowdown}")

    """ output = policy_module(reset_env)
    action = output["action"]
    print(f"action: {action}")

    next_state, reward, done, info = env.step(output)
    print(f"reward: {reward}, info: {info}") """
