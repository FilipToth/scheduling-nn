import environments.env as environments

env = environments.MachineEnvironment()
initial_state, _ = env.reset()
print(f'initial state shape: {initial_state.shape}')

while True:
    env.render()

    print(env._resources)
    action = int(input("action -> "))
    observation, reward, terminated, truncated, _ = env.step(action)
