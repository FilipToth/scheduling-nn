import environments.env as env

environment = env.MachineEnvironment()
initial_state, _ = environment.reset()
print(f'initial state shape: {initial_state.shape}')

while True:
    environment.render()

    print(environment._resources)
    action = int(input("action -> "))
    observation, reward, terminated, truncated, _ = environment.step(action)