from __future__ import annotations

import enum
import random
import numpy as np
import gymnasium as gym

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf

# size of the job buffer
JOB_QUEUE_SIZE = 4
NUM_RESOURCES = 2
RESOURCE_TIME_BUFFER_SIZE = 4

class ResourceType(enum.Enum):
    CPU = 0
    MEMORY = 1


class Job:
    empty = False
    scheduled: bool = False
    time_scheduled: int | None = None

    def __init__(self, resource_use: dict[ResourceType, float], time_use: int, time_dispatched: int) -> None:
        self.resource_use = resource_use
        self.time_use = time_use
        self.time_dispatched = time_dispatched

    def schedule(self, time: int) -> None:
        self.scheduled = True
        self.time_scheduled = time

    def expected_job_slowdown(self) -> int:
        if not self.scheduled:
            return 0

        waiting_time = self.time_scheduled - self.time_dispatched
        service_time = self.time_use

        slowdown = (waiting_time + service_time) / service_time
        return slowdown

    def new_random(curr_time: int) -> Job:
        resources = {}
        for res in ResourceType:
            use = random.uniform(0, 1)
            resources[res] = use

        time_use = random.randint(1, RESOURCE_TIME_BUFFER_SIZE)

        job = Job(resources, time_use, curr_time)
        return job

    def new_empty(curr_time: int = 0) -> Job:
        resources = {}
        for res in ResourceType:
            resources[res] = 0.0

        job = Job(resources, 0, curr_time)
        job.empty = True

        return job


class MachineEnvironment(gym.Env):
    def __init__(self) -> None:
        # note that jobs also
        # have a timeframe,
        # thus the + 1
        self._job_queue: list[Job] = []
        self._running_jobs: list[Job] = []
        self._resources = {}
        self._time = 0

        # + 1 since the AI can
        # also choose to do nothing
        self.action_space = gym.spaces.Discrete(JOB_QUEUE_SIZE + 1)

    def step(self, action: int):
        # process action
        if action == JOB_QUEUE_SIZE:
            # skipping will have no effect
            # on the resource schedule, thus
            # we can also skip the timeframe
            self.time_step()

            # TODO: Calculate job slowdown and return reward

            reward = 0
            for job in self._job_queue:
                if job.time_use == 0:
                    continue

                reward += -1 / job.time_use

            return self._get_obs(), reward, False, False, self._get_info()
        else:
            job = self._job_queue[action]
            if job.empty:
                # empty job slot
                return self._get_obs(), 0, False, False, self._get_info()

            job_resources = job.resource_use
            res_with_job = self._resources.copy()
            for res, am in job_resources.items():
                res_with_job[res] += am

            is_not_over = all(r <= 1.0 for r in res_with_job.values())
            if not is_not_over:
                # the scheduler allocated too many jobs
                # and now a resource is overused
                return self._get_obs(), 0, False, False, self._get_info()

            # remove job from queue
            self._job_queue[action] = Job.new_empty()

            # update machine resources
            self._resources = res_with_job

            # schedule job
            job.schedule(self._time)
            self._running_jobs.append(job)

            terminated = all(j.empty for j in self._job_queue)
            state = self._get_obs()
            info = self._get_info()

            return state, 0, terminated, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._job_queue = []
        self._running_jobs = []
        self._resources = {}
        self._time = 0

        self.time_step(is_init=False)

        # generate job queue
        for _ in range(JOB_QUEUE_SIZE):
            job = Job.new_random(self._time)
            self._job_queue.append(job)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def render(self):
        self._render_frame()

    def _render_frame(self):
        pass

    def time_step(self, is_init = False):
        if not is_init:
            self._time += 1

        self._step_resource_use()
        self._step_scheduled_jobs()

    def _step_resource_use(self):
        self._resources = {}
        for res in ResourceType:
            self._resources[res] = 0.0

    def _step_scheduled_jobs(self):
        jobs_to_unschedule = []
        for job in self._running_jobs:
            time_done = job.time_scheduled + job.time_use
            if not (time_done == self._time):
                # add job resources to res use
                for res, am in job.resource_use.items():
                    self._resources[res] += am

                continue

            # job is done
            jobs_to_unschedule.append(job)

        for job in jobs_to_unschedule:
            self._running_jobs.remove(job)

    def _get_obs(self):
        state = []
        for res_used in self._resources.values():
            state.append(res_used)

        for job in self._job_queue:
            assert not job.scheduled
            state.extend(job.resource_use.values())

        state = np.array(state, dtype=np.float32)

        return state

    def _get_info(self):
        return {}

    def close(self):
        return super().close()
