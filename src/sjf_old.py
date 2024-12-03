import math
import random
import environments.env as env

rewards = []

environment = env.MachineEnvironment()
environment.reset()

cumulative_reward = 0
for _ in range(50):
    environment.step(0)
    continue

    action = 0
    if len(environment.job_queue) > 0:
        shortest_job_index = 0
        shortest_job_time = math.inf

        for index, job in enumerate(environment.job_queue):
            time = job.time_use
            if shortest_job_time <= time:
                continue

            shortest_job_index = index
            shortest_job_time = time

        action = shortest_job_index

    _observation, reward, terminated, _, info = environment.step(action)
    print(action)
    cumulative_reward += reward
    exit()

    if terminated:
        break

slowdown = environment.mean_slowdown()
print(f'cumulative reward: {cumulative_reward}, slowdown: {slowdown}')
