import tkinter
import numpy as np
from torchrl.envs.libs.gym import GymEnv
from environments.env import ACTION_SPACE_SIZE, NUM_RESOURCES, RESOURCE_TIME_SIZE

WIDTH = (NUM_RESOURCES * 90) + (ACTION_SPACE_SIZE * 240 * NUM_RESOURCES) + 100
HEIGHT = (90 * RESOURCE_TIME_SIZE)

MATRIX_SPACING = 10
MATRIX_DOT_SIZE = 50

class EnvironmentVisualization:
    def __init__(self, env: GymEnv, action_callback) -> None:
        self.env = env
        self.action_callback = action_callback

        root = tkinter.Tk()
        root.geometry(f"{WIDTH}x{HEIGHT + 100}")

        self.canvas = tkinter.Canvas(root, width=WIDTH, height=HEIGHT)
        self.canvas.pack()

        btn = tkinter.Button(root, text="AI Step", command=lambda: self.btn_click())
        btn.pack()

        self.draw_env()
        self.canvas.mainloop()

    def draw_env(self):
        self.canvas.delete("all")

        # resources have a shape (NUM_RES, TIME_HORIZONT)
        # draw resource state space
        self.draw_matrix(self.env.resources, 30, 30)

        for index, job in enumerate(self.env.job_queue):
            state = []
            for _ in range(job.time_use):
                time_state = []
                for (_, use) in job.resource_use.items():
                    time_state.append(use)

                state.append(time_state)

            nda = np.array(state).transpose()
            self.draw_matrix(nda, 200 + (index * 150), 30)

    def draw_matrix(self, matrix: np.ndarray, origin_x: int, origin_y: int):
        for (j, k), value in np.ndenumerate(matrix):
            x = origin_x + j * (MATRIX_SPACING + MATRIX_DOT_SIZE)
            y = origin_y + k * (MATRIX_SPACING + MATRIX_DOT_SIZE)

            x1 = x + MATRIX_DOT_SIZE
            y1 = y + MATRIX_DOT_SIZE
            y_fill = y + (MATRIX_DOT_SIZE * value)

            self.canvas.create_rectangle(x, y, x1, y_fill, fill="blue")
            self.canvas.create_rectangle(x, y, x1, y1, width=2)

    def btn_click(self):
        self.action_callback()
        self.draw_env()
