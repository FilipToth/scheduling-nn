from tensordict import TensorDict
from ppo.consume import init_consume
from viz import EnvironmentVisualization

policy_module, env, base_env = init_consume("../out/ppo/policy_module.pth")
latest_obs = env.reset()
env.reset_logs()

def action_callback():
    global latest_obs

    model_out = policy_module(latest_obs)
    action = model_out["action"]

    action_tensordict = TensorDict({
        "action": action,
        "step_count": 1,
    }, batch_size=env.batch_size)

    env_out = env.step(action_tensordict)
    latest_obs = env_out["next"]


_viz = EnvironmentVisualization(base_env, action_callback, True)
