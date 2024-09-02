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

    def __init__(self, resource_use: dict[ResourceType, float], time: int) -> None:
        self.resource_use = resource_use
        self.scheduled = False

    def new_random() -> Job:
        resources = {}
        for res in ResourceType:
            use = random.uniform(0, 1)
            resources[res] = use

        time = random.randint(1, RESOURCE_TIME_BUFFER_SIZE)

        job = Job(resources, time)
        return job

    def new_empty() -> Job:
        resources = {}
        for res in ResourceType:
            resources[res] = 0.0

        job = Job(resources, 0)
        job.empty = True

        return job


class MachineEnvironment(gym.Env):
    def __init__(self) -> None:
        # note that jobs also
        # have a timeframe,
        # thus the + 1
        self._job_queue: list[Job] = []
        self._running_jobs = []
        self._resources = {}

        # + 1 since the AI can
        # also choose to do nothing
        self.action_space = gym.spaces.Discrete(JOB_QUEUE_SIZE + 1)

    def step(self, action: int):
        # so far we don't have the concept
        # of time, this will only happen on
        # each timestep, resource use will
        # also be a time matrix not a vector
        self._step_resource_use()

        # process action
        if action == JOB_QUEUE_SIZE:
            # skip
            return self._get_obs(), -0.1, False, False, self._get_info()
        else:
            job = self._job_queue[action]
            if job.empty:
                # empty job slot
                return self._get_obs(), -0.1, False, False, self._get_info()

            job_resources = job.resource_use
            res_with_job = self._resources.copy()
            for res, am in job_resources.items():
                res_with_job[res] += am

            is_not_over = all(r <= 1.0 for r in res_with_job.values())
            if not is_not_over:
                # the scheduler allocated too many jobs
                # and now a resource is overused
                return self._get_obs(), -0.2, False, False, self._get_info()

            # remove job from queue
            self._job_queue[action] = Job.new_empty()

            # update machine resources
            self._resources = res_with_job

        terminated = all(j.empty for j in self._job_queue)
        state = self._get_obs()
        info = self._get_info()

        return state, 1, terminated, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._job_queue = []
        self._resources = {}
        self._step_resource_use()

        # generate job queue
        for _ in range(JOB_QUEUE_SIZE):
            job = Job.new_random()
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

    def _step_resource_use(self):
        self._resources = {}
        for res in ResourceType:
            self._resources[res] = 0.0

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
