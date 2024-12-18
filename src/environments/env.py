from __future__ import annotations

import enum
import torch
import random
import numpy as np
import gymnasium as gym

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf

# size of the job buffer
ACTION_SPACE_SIZE = 4
NUM_RESOURCES = 2
RESOURCE_TIME_SIZE = 10

NUM_JOBS_TO_SEND = 80
# JOB_DISPATCH_AFFINITY = 0.4
MAX_JOB_LENGTH = 6

class ResourceType(enum.Enum):
    CPU = 0
    MEMORY = 1


class Job:
    scheduled: bool = False
    time_scheduled: int | None = None

    def __init__(self, resource_use: dict[ResourceType, float], time_use: int, time_dispatched: int) -> None:
        self.resource_use = resource_use
        self.time_use = time_use
        self.time_dispatched = time_dispatched

    def schedule(self, time: int) -> None:
        self.scheduled = True
        self.time_scheduled = time

    def expected_job_slowdown(self) -> float:
        if not self.scheduled:
            return 0.0

        waiting_time = self.time_scheduled - self.time_dispatched
        service_time = self.time_use

        slowdown = (waiting_time + service_time) / service_time
        return slowdown


class MachineEnvironment(gym.Env):
    def __init__(self, log_path = None) -> None:
        # note that jobs also
        # have a timeframe,
        # thus the + 1
        self.job_queue: list[Job] = []
        self._scheduled_jobs: list[Job] = []
        self._finished_jobs: list[Job] = []
        self.resources: np.ndarray = None
        self.time = 0
        self._jobs_dispatched = 0
        self.seed_value = None

        if log_path != None:
            self.log_handle = open(log_path, "w+")
            self.reset_logs()
        else:
            self.log_handle = None

        # + 1 since the AI can
        # also choose to do nothing
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE + 1)

        # + 1 because it jobs also have a time parameter
        job_queue_state_size = ACTION_SPACE_SIZE * (NUM_RESOURCES + 1)
        resource_state_size = NUM_RESOURCES * RESOURCE_TIME_SIZE

        obs_space_size = job_queue_state_size + resource_state_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_space_size,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        # TODO: Passing seeds is not working for some reason
        # this is a quick fix...
        seed = 5051
        if seed is None:
            return

        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return [ seed ]

    def get_time(self):
        return self.time

    def step(self, action: int):
        # process action
        reward = 0
        info = ""

        if action == ACTION_SPACE_SIZE:
            self.time_step()
            info = "TIMESTEP"
        else:
            if action >= len(self.job_queue):
                # empty job slot, also time step
                self.time_step()
                info = "EMPTY"
            else:
                job = self.job_queue[action]
                (success_alloc, time_frame) = self._alloc(job)
                if not success_alloc:
                    # the scheduler allocated too many jobs
                    # and now a resource is overused, time step
                    self.time_step()
                    info = "OVERALLOC"
                else:
                    # remove job from queue
                    del self.job_queue[action]

                    # schedule job
                    job.schedule(self.time + time_frame)
                    self._scheduled_jobs.append(job)
                    info = "ALLOC"
                    reward = 0.25

        if info != "ALLOC":
            slowdown = self.mean_slowdown()
            if slowdown == 0.0:
                slowdown = 1

            reward = -slowdown

        state = self._get_obs()
        terminated = self._jobs_dispatched >= NUM_JOBS_TO_SEND \
            and len(self.job_queue) == 0 \
            and len(self._scheduled_jobs) == 0

        self._log(f"action: {action}, terminated: {terminated}, js_dispatched: {self._jobs_dispatched}, j_queue: {len(self.job_queue)}, sched_jobs: {len(self._scheduled_jobs)}")
        return state, reward, terminated, False, info

    def reset(self, *, _options=None):
        slowdown = self.mean_slowdown()
        self._log(f"Resetting; slowdown: {slowdown}")
        super().reset()
        # self.seed(self.seed_value)

        self.job_queue = []
        self._finished_jobs: list[Job] = []
        self._scheduled_jobs = []
        self.resources = np.zeros((NUM_RESOURCES, RESOURCE_TIME_SIZE))
        self.time = 0
        self._jobs_dispatched = 0


        # TODO: is this really necessary?
        self.time_step(is_init=True)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def render(self):
        self._render_frame()

    def _render_frame(self):
        pass

    def time_step(self, is_init=False):
        if not is_init:
            self.time += 1

        self._job_dispatch()
        self._step_resource_use()
        self._step_scheduled_jobs()

    def mean_slowdown(self) -> float:
        jobs = self._finished_jobs + self._scheduled_jobs
        if len(jobs) == 0:
            return -1

        sum_slowdown = 0.0
        for job in jobs:
            sum_slowdown += job.expected_job_slowdown()

        mean_slowdown = sum_slowdown / len(jobs)
        return mean_slowdown

    def _step_resource_use(self):
        res = np.delete(self.resources, (0), axis=1)
        new_row = np.zeros(shape=(NUM_RESOURCES, 1))

        res = np.append(res, new_row, axis=1)
        self.resources = res

    def _step_scheduled_jobs(self):
        jobs_to_unschedule = []
        for job in self._scheduled_jobs:
            time_done = job.time_scheduled + job.time_use
            if time_done > self.time:
                continue

            # job is done
            jobs_to_unschedule.append(job)

        for job in jobs_to_unschedule:
            self._finished_jobs.append(job)
            self._scheduled_jobs.remove(job)

    def _job_dispatch(self):
        if self._jobs_dispatched >= NUM_JOBS_TO_SEND:
            return

        jobs_queued = len(self.job_queue)
        if jobs_queued > ACTION_SPACE_SIZE:
            return

        jobs_to_send = ACTION_SPACE_SIZE - jobs_queued
        for _ in range(jobs_to_send):
            if self._jobs_dispatched >= NUM_JOBS_TO_SEND:
                # break
                return

            resources = {}
            for res in ResourceType:
                res_use = random.uniform(0, 1)
                resources[res] = res_use

            time_use = random.randrange(1, MAX_JOB_LENGTH + 1)
            job = Job(resources, time_use, self.time)

            self.job_queue.append(job)
            self._jobs_dispatched += 1

    def _alloc(self, job: Job) -> tuple[bool, int]:
        job_resources = list(job.resource_use.values())

        for time_start in range(RESOURCE_TIME_SIZE - job.time_use):
            timeframe_start = time_start
            timeframe_end = time_start + job.time_use

            job_resource_matrix = np.zeros(shape=(NUM_RESOURCES, job.time_use))
            for i in range(job.time_use):
                for j in range(len(job_resources)):
                    res = job_resources[j]
                    job_resource_matrix[j:j + 1, i:i + 1] = res

            local_time_resource = self.resources[:, time_start:timeframe_end]
            local_resources_with_job = np.add(local_time_resource, job_resource_matrix)

            success = np.all(local_resources_with_job <= 1)
            if not success:
                continue

            # update machine resources
            self.resources[:, timeframe_start:timeframe_end] = local_resources_with_job
            return (True, time_start)

        return (False, 0)

    def _get_obs(self):
        state = self.resources.flatten()

        # append job queue info
        job_state = []

        queue_len = min(ACTION_SPACE_SIZE, len(self.job_queue))
        for job in self.job_queue[0:queue_len]:
            assert not job.scheduled

            job_state.extend(job.resource_use.values())
            job_state.append(job.time_use)

        # pad to action space size
        if queue_len < ACTION_SPACE_SIZE:
            to_add = ACTION_SPACE_SIZE - queue_len
            for _ in range(to_add):
                job_state.extend([0 for _ in range(NUM_RESOURCES)])
                job_state.append(0)

        job_state = np.array(job_state)
        state = np.concatenate((state, job_state))
        return state

    def _get_info(self):
        return {}

    def _log(self, msg):
        if self.log_handle == None:
            return

        self.log_handle.write(msg + "\n")

    def reset_logs(self):
         self.log_handle.truncate(0)

    def close(self):
        if self.log_handle != None:
            self.log_handle.close()

        return super().close()
