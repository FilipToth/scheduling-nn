
from viz import EnvironmentVisualization
from ppo.consume import init_consume

policy_module, env, base_env = init_consume()
initial = env.reset()

def action_callback():
    env.rollout(max_steps=1, policy=policy_module)


_viz = EnvironmentVisualization(base_env, action_callback)
