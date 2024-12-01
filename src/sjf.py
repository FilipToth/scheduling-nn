import math
import random
import environments.env as env

NUM_TRIES = 1000

rewards = []

for i in range(NUM_TRIES):
    environment = env.MachineEnvironment()
    seed = random.randint(0, 100_000_000)
    environment.reset(seed=seed)

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

    rewards.append(cumulative_reward)
    print(f'[{i}/{NUM_TRIES}] {seed} cumulative reward: {cumulative_reward}')

mean_reward = sum(rewards) / len(rewards)
print(f'mean reward: {mean_reward}')
