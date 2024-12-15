import torch
from ppo.consume import init_consume

policy_module, env, base_env = init_consume("../out/ppo/policy_module.pth")

with torch.no_grad():
    print("EVALUATING MODEL...")

    initial = env.reset()
    print(initial["observation"])
    rollout = env.rollout(max_steps=1000, policy=policy_module, tensordict=initial, auto_reset=False)

    slowdown = base_env.mean_slowdown()
    time = base_env.time

    print(f"mean slowdown: {slowdown}, time: {time}")

    """ output = policy_module(reset_env)
    action = output["action"]
    print(f"action: {action}")

    next_state, reward, done, info = env.step(output)
    print(f"reward: {reward}, info: {info}") """
