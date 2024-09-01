import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf

# size of the job buffer
JOB_QUEUE_SIZE = 9
NUM_RESOURCES = 2
RESOURCE_TIME_BUFFER_SIZE = 9

class MachineEnvironment(gym.Env):
    def __init__(self) -> None:
        # note that jobs also
        # have a timeframe,
        # thus the + 1
        self._resources = np.zeros(shape=(NUM_RESOURCES, RESOURCE_TIME_BUFFER_SIZE), dtype=np.float32)
        self._job_queue = np.zeros(shape=(NUM_RESOURCES + 1, JOB_QUEUE_SIZE), dtype=np.float32)

        # + 1 since the AI can
        # also choose to do nothing
        self.action_space = gym.spaces.Discrete(JOB_QUEUE_SIZE + 1)

    def step(self, action):
        # progress the resource matrix one step in time
        # print(self._resources)
        resources = self._resources
        resources = np.delete(resources, (0), axis=1)

        new_res_row = np.zeros(shape=(NUM_RESOURCES, 1), dtype=np.float32)
        resources = np.append(resources, new_res_row, axis=1)

        # process action
        if action == JOB_QUEUE_SIZE:
            # skip
            return self._get_obs(), -0.1, False, False, self._get_info()
        else:
            job = self._job_queue[:, action:action+1]
            if np.all(job == 0.0):
                # empty job slot
                return self._get_obs(), -0.1, False, False, self._get_info()

            time_frame = job[NUM_RESOURCES]
            job_resources = job[0:NUM_RESOURCES, 0:1]
            res_with_job = resources[0:NUM_RESOURCES, 0:1] + job_resources

            is_not_over = np.all(res_with_job <= 1.0)
            if not is_not_over:
                # the scheduler allocated too many jobs
                # and now a resource is overused
                return self._get_obs(), -0.2, False, False, self._get_info()

            # remove job from queue
            self._job_queue[:, action] = 0

            # update machine resources
            resources[0:NUM_RESOURCES, 0:1] = res_with_job
            self._resources = resources

        terminated = False
        if np.all(self._job_queue == 0.0):
            terminated = True

        return self._get_obs(), 1, terminated, False, self._get_info()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._resources = np.zeros(shape=(NUM_RESOURCES, RESOURCE_TIME_BUFFER_SIZE), dtype=np.float32)
        self._job_queue = np.random.random(size=(NUM_RESOURCES + 1, JOB_QUEUE_SIZE))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def render(self):
        self._render_frame()

    def _render_frame(self):
        pass

    def _get_obs(self):
        resources = self._resources.flatten()
        job_queue = self._job_queue.flatten()

        # print(f'_get_obs, resources shape: {resources.shape}')
        # print(f'_get_obs, job_queue shape: {job_queue.shape}')

        state = np.concatenate((resources, job_queue))
        return state

    def _get_info(self):
        return {}

    def close(self):
        return super().close()