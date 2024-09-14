import unittest
import numpy as np
import src.environments.env as env

class TestEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.env = env.MachineEnvironment()

        # don't use the reset function,
        # as we want to use custom overrides

        self.env._job_queue = []
        self.env._scheduled_jobs = []
        self.env._resources = np.zeros((env.NUM_RESOURCES, env.RESOURCE_TIME_SIZE))
        self.env._time = 0

        self.env.time_step(is_init=True)

    def test_simple_alloc(self):
        job_len = 4
        job_res = 0.9
        job = self._create_job(job_len, job_res)

        assert np.all(self.env._resources == 0.0)

        (success, alloc_time) = self.env._alloc(job)
        assert success
        assert alloc_time == 0

        res_for_job_timeframe = self.env._resources[:, 0:job_len]
        assert np.all(res_for_job_timeframe == 0.9)

    def test_alloc_multiple_jobs(self):
        first_job_len = 4
        first_job_res = 0.9
        first_job = self._create_job(first_job_len, first_job_res)

        (first_success, first_alloc_time) = self.env._alloc(first_job)
        assert first_success
        assert first_alloc_time == 0

        second_job_len = 2
        second_job_res = 0.5
        second_job = self._create_job(second_job_len, second_job_res)

        (second_success, second_alloc_time) = self.env._alloc(second_job)
        assert second_success
        assert second_alloc_time == first_job_len

    def test_alloc_multiple_jobs_same_timeframe(self):
        first_job_len = 4
        first_job_res = 0.3
        first_job = self._create_job(first_job_len, first_job_res)

        (first_success, first_alloc_time) = self.env._alloc(first_job)
        assert first_success
        assert first_alloc_time == 0

        second_job_len = 2
        second_job_res = 0.5
        second_job = self._create_job(second_job_len, second_job_res)

        (second_success, second_alloc_time) = self.env._alloc(second_job)
        assert second_success
        assert second_alloc_time == 0

        third_job = self._create_job(second_job_len, second_job_res)
        (third_success, third_alloc_time) = self.env._alloc(third_job)

        assert third_success
        assert third_alloc_time == second_job_len

        fourth_job = self._create_job(second_job_len, second_job_res)
        (fourth_success, fourth_alloc_time) = self.env._alloc(fourth_job)

        assert fourth_success
        assert fourth_alloc_time == first_job_len

    def test_alloc_fails(self):
        first_job_len = 9
        first_job_res = 0.8
        first_job = self._create_job(first_job_len, first_job_res)

        (first_success, first_alloc_time) = self.env._alloc(first_job)
        assert first_success
        assert first_alloc_time == 0

        second_job_len = 2
        second_job_res = 0.4
        second_job = self._create_job(second_job_len, second_job_res)

        (second_success, second_alloc_time) = self.env._alloc(second_job)
        assert not second_success
        assert second_alloc_time == 0

    def _create_job(self, time: int, res: float) -> env.Job:
        job_res = {
            env.ResourceType.CPU: res,
            env.ResourceType.MEMORY: res
        }

        job = env.Job(job_res, time, self.env._time)
        return job
