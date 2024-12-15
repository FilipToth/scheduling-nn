import torch
from gymnasium import register
from torch import multiprocessing
from ppo.models import setup_model
from torchrl.envs.libs.gym import GymEnv
from environments.env import MachineEnvironment
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

def init_consume(policy_path: str) -> tuple[TensorDictModule, TransformedEnv, GymEnv]:
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

    base_env = GymEnv("MachineEnv-v0", device=device, log_path="../out/consume.log")

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
    env.set_seed(5051)

    policy_module, _ = setup_model(env)
    policy_module = policy_module.to(device)

    policy_net_state = torch.load(policy_path)
    policy_module.load_state_dict(policy_net_state)

    policy_module(env.reset())
    policy_module.eval()

    return policy_module, env, base_env
