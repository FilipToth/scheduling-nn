import torch
from gymnasium import register
from torch import multiprocessing
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from environments.env import MachineEnvironment
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

register(
    id="MachineEnv-v0",
    entry_point=MachineEnvironment
)

base_env = GymEnv("MachineEnv-v0", device=device)

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.reset()

action = torch.tensor([0])
action_tensor_dict = TensorDict({ "action": action, "step_count": torch.tensor(0) }, batch_size=())

result = env.step(action_tensor_dict)
print(result)

cumulative_reward = 0
print(base_env.job_queue)
