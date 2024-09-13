from __future__ import annotations

import enum
import random
import numpy as np
import gymnasium as gym

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf

# size of the job buffer
ACTION_SPACE_SIZE = 4
JOB_QUEUE_SIZE = 30
NUM_RESOURCES = 2
RESOURCE_TIME_SIZE = 10
MAX_JOB_LENGTH = 4

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

        time_use = random.randint(1, MAX_JOB_LENGTH,)

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
        self._resources: np.ndarray = None
        self._time = 0

        # + 1 since the AI can
        # also choose to do nothing
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE + 1)

    def step(self, action: int):
        # process action
        if action == ACTION_SPACE_SIZE:
            reward = self.timestep_action()
            return self._get_obs(), reward, False, False, "TIMESTEP"
        else:
            job = self._job_queue[action]
            if job.empty:
                # empty job slot, also time step
                reward = self.timestep_action()
                return self._get_obs(), reward, False, False, "EMPTY"

            success_alloc = self._alloc(job)
            if not success_alloc:
                # the scheduler allocated too many jobs
                # and now a resource is overused, time step
                reward = self.timestep_action()
                return self._get_obs(), reward, False, False, "OVERALLOC"

            # remove job from queue
            del self._job_queue[action]
            if len(self._job_queue) < ACTION_SPACE_SIZE:
                empty_job = Job.new_empty()
                self._job_queue.append(empty_job)

            # schedule job
            job.schedule(self._time)
            self._running_jobs.append(job)

            terminated = all(j.empty for j in self._job_queue)
            state = self._get_obs()

            return state, 0, terminated, False, "ALLOC"

    def timestep_action(self):
        self.time_step()

        # TODO: Calculate job slowdown and return reward

        reward = 0
        for job in self._job_queue:
            if job.time_use == 0:
                continue

            reward += -1 / job.time_use

        return reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)

        self._job_queue = []
        self._running_jobs = []
        self._resources = np.zeros((NUM_RESOURCES, RESOURCE_TIME_SIZE))
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
        res = np.delete(self._resources, (0), axis=1)
        new_row = np.zeros(shape=(NUM_RESOURCES, 1))

        res = np.append(res, new_row, axis=1)
        self._resources = res

    def _step_scheduled_jobs(self):
        jobs_to_unschedule = []
        for job in self._running_jobs:
            time_done = job.time_scheduled + job.time_use
            if not (time_done == self._time):
                continue

            # job is dones
            jobs_to_unschedule.append(job)

        for job in jobs_to_unschedule:
            self._running_jobs.remove(job)

    def _alloc(self, job: Job):
        job_resources = list(job.resource_use.values())

        for time_start in range(RESOURCE_TIME_SIZE - job.time_use):
            timeframe_start = time_start
            timeframe_end = time_start + job.time_use

            job_resource_matrix = np.zeros(shape=(NUM_RESOURCES, job.time_use))
            for i in range(timeframe_start, timeframe_end):
                for j in range(len(job_resources)):
                    res = job_resources[j]
                    job_resource_matrix[j:j + 1, i:i + 1] = res

            local_time_resource = self._resources[:, time_start:timeframe_end]
            local_resources_with_job = np.add(local_time_resource, job_resource_matrix)

            success = np.all(local_resources_with_job <= 1)
            if not success:
                continue

            # update machine resources
            self._resources[:, timeframe_start:timeframe_end] = local_resources_with_job
            return True

        return False

    def _get_obs(self):
        state = self._resources.flatten()

        # append job queue info
        job_state = []
        for job in self._job_queue[0:ACTION_SPACE_SIZE]:
            assert not job.scheduled

            job_state.extend(job.resource_use.values())
            job_state.append(job.time_use)

        job_state = np.array(job_state)
        state = np.concatenate((state, job_state))
        return state

    def _get_info(self):
        return {}

    def close(self):
        return super().close()
