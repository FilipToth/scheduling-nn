from torch import nn
from torchrl.envs import EnvBase
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

NUM_CELLS = 256

def setup_model(env: EnvBase):
    actor_net = nn.Sequential(
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1]),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": 0,
            "max": env.action_space.n - 1,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = nn.Sequential(
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS),
        nn.Tanh(),
        nn.LazyLinear(1),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return policy_module, value_module
