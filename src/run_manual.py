from viz import EnvironmentVisualization
from environments.env import MachineEnvironment

env = MachineEnvironment()
env.reset()

def action_callback():
    print()

    action = int(input("action? "))
    env.step(action)


_viz = EnvironmentVisualization(env, action_callback, False)
