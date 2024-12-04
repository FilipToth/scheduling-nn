import math
import random
import environments.env as env

rewards = []

environment = env.MachineEnvironment()
environment.reset(seed=5051)

cumulative_reward = 0
for i in range(5000):
    # print(len(environment.job_queue))
    action = 4
    if len(environment.job_queue) > 0:
        # select shortest job
        shortest_job_index = 0
        shortest_job_time = math.inf

        for index, job in enumerate(environment.job_queue):
            time = job.time_use
            if shortest_job_time <= time:
                continue

            shortest_job_index = index
            shortest_job_time = time

        action = shortest_job_index

    state, reward, terminated, _, info = environment.step(action)
    cumulative_reward += reward

    if terminated:
        break

slowdown = environment.mean_slowdown()
print(f'cumulative reward: {cumulative_reward}, slowdown: {slowdown}')
