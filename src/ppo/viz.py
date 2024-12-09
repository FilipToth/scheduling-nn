import torch
import tkinter
import numpy as np
from ppo.consume import init_consume
from environments.env import ACTION_SPACE_SIZE, NUM_RESOURCES, RESOURCE_TIME_SIZE

WIDTH = (NUM_RESOURCES * 90) + (ACTION_SPACE_SIZE * 240 * NUM_RESOURCES) + 100
HEIGHT = (90 * RESOURCE_TIME_SIZE)

MATRIX_SPACING = 10
MATRIX_DOT_SIZE = 50

root = tkinter.Tk()
root.geometry(f"{WIDTH}x{HEIGHT + 100}")

canvas = tkinter.Canvas(root, width=WIDTH, height=HEIGHT)
canvas.pack()

policy_module, env, base_env = init_consume()

def draw_env():
    canvas.delete("all")

    # resources have a shape (NUM_RES, TIME_HORIZONT)
    # draw resource state space
    draw_matrix(env.resources, 30, 30)

    for index, job in enumerate(env.job_queue):
        state = []
        for t in range(job.time_use):
            time_state = []
            for (_, use) in job.resource_use.items():
                time_state.append(use)

            state.append(time_state)

        nda = np.array(state).transpose()
        draw_matrix(nda, 200 + (index * 150), 30)

def draw_matrix(matrix: np.ndarray, origin_x: int, origin_y: int):
    for (j, k), value in np.ndenumerate(matrix):
        x = origin_x + j * (MATRIX_SPACING + MATRIX_DOT_SIZE)
        y = origin_y + k * (MATRIX_SPACING + MATRIX_DOT_SIZE)

        x1 = x + MATRIX_DOT_SIZE
        y1 = y + MATRIX_DOT_SIZE
        y_fill = y + (MATRIX_DOT_SIZE * value)

        canvas.create_rectangle(x, y, x1, y_fill, fill="blue")
        canvas.create_rectangle(x, y, x1, y1, width=2)


initial = env.reset()
def btn_click():
    env.rollout(max_steps=1, policy=policy_module)
    draw_env()

btn = tkinter.Button(root, text="AI Step", command=btn_click)
btn.pack()

draw_env()
canvas.mainloop()