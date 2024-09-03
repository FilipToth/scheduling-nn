import environments.env as env

environment = env.MachineEnvironment()
initial_state, _ = environment.reset()
print(f'initial state shape: {initial_state.shape}')

while True:
    environment.render()

    action = int(input())
    observation, reward, terminated, truncated, _ = environment.step(action)
    # print(f'observation shape: {observation.shape}')
    # print(f'reward: {reward}')
