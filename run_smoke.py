import numpy as np
from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file=r'C:\Users\USER\Desktop\SUMORL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml',
    route_file=r'C:\Users\USER\Desktop\SUMORL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml',
    out_csv_name='smoke_results.csv',
    use_gui=False,
    num_seconds=30,
    single_agent=True,
    min_green=5,
    max_green=50,
)

state = env.reset()[0]
finished = False
steps = 0
reward = 0
while not finished and steps < 500:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    finished = terminated or truncated
    steps += 1

print(f"Smoke run finished after {steps} steps; last reward: {reward}")
env.close()
