import math
import environments.env as env

environment = env.MachineEnvironment()
environment.reset(seed=93021)

cumulative_reward = 0
while True:
    action = 0
    if len(environment._job_queue) > 0:
        shortest_job_index = 0
        shortest_job_time = math.inf

        for index, job in enumerate(environment._job_queue):
            time = job.time_use
            if shortest_job_time <= time:
                continue

            shortest_job_index = index
            shortest_job_time = time

        action = shortest_job_index

    _observation, reward, terminated, _, _ = environment.step(action)
    cumulative_reward += reward

    if terminated:
        break

print(f'cumulative reward: {cumulative_reward}')