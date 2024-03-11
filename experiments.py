import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
import cProfile
from time import perf_counter
from planners import RRT_STAR
from pprint import pprint
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def init():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)
    
    bb = Building_Blocks(transform=transform, 
                        ur_params=ur_params, 
                        env=env,
                        resolution=0.1, 
                        p_bias=0,)
    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
    return bb, visualizer

def test(bb: Building_Blocks):
    for i in range(10000):
        bb.is_in_collision(bb.sample([]))


def run_trials(p_bias, max_step_size, num_trials) -> tuple[float, int, float]:
    """
    Run a number of trials and save the results to a csv file
    """
    ur_params = UR5e_PARAMS(inflation_factor=1)
    bb = Building_Blocks(Transform(ur_params), ur_params, Environment(2), 0.1, p_bias)
    start_conf = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    goal_conf = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    results = []
    for i in range(num_trials):
        planner = RRT_STAR(max_step_size, 2000, bb)
        start = perf_counter()
        _, cost = planner.find_path(start_conf, goal_conf, "", True)
        end = perf_counter()
        time_taken = end - start
        print(f"in proccess {os.getpid()}:")
        print(f"trial {i} took {time_taken} seconds")
        print(f"cost: {cost}")
        results.append((p_bias, max_step_size, time_taken, cost != np.inf, cost))
    df = pd.DataFrame(results, columns=["p_bias", "max_step_size", "time", "success", "cost"])
    df.to_csv('results.csv', mode='a', header=False, index=False)


def compute_parameter_matrix(p_bias: list[float], max_step_size: list[float], num_trials: int = 10):
    for bias in p_bias:
        for step_size in max_step_size:
            run_trials(bias, step_size, num_trials)

def compute_matrix_results(filename='results.csv'):
    """
    Compute the mean and standard deviation of the time, the ratio of successful trials to all trials and the mean cost
    """
    df = pd.read_csv(filename)
    df_grouped = df.groupby(["p_bias", "max_step_size"]).agg({"time": ["mean", "std"], "success": "mean"})
    df_grouped['cost'] = df[df['cost'].apply(np.isfinite)].groupby(["p_bias", "max_step_size"])['cost'].mean()
    df_grouped['trials'] = df.groupby(["p_bias", "max_step_size"]).size()

    # df = df.groupby(["p_bias", "max_step_size"]).agg({"time": ["mean", "std"], "success": "mean", "cost": "mean"})
    print(df_grouped)
    plot_results(df_grouped)

def plot_results(df_grouped):
    # Pivot the DataFrame to create a matrix for each variable
    time_matrix = df_grouped[('time', 'mean')].unstack()
    success_matrix = df_grouped[('success', 'mean')].unstack()
    cost_matrix = df_grouped['cost'].unstack()

    # Create a heatmap for the mean time
    plt.figure(figsize=(8, 6))
    sns.heatmap(time_matrix, annot=True, fmt=".2f", cmap='viridis_r')
    plt.title("Mean Time for Each Parameter Combination")
    plt.xlabel("Max Step Size")
    plt.ylabel("P Bias")
    plt.savefig('imgs/time_heatmap.png')
    plt.close()

    # Create a heatmap for the success rate
    plt.figure(figsize=(8, 6))
    sns.heatmap(success_matrix, annot=True, fmt=".2f", cmap='viridis')
    plt.title("Success Rate for Each Parameter Combination")
    plt.xlabel("Max Step Size")
    plt.ylabel("P Bias")
    plt.savefig('imgs/success_rate_heatmap.png')
    plt.close()

    # Create a heatmap for the mean cost
    plt.figure(figsize=(8, 6))
    sns.heatmap(cost_matrix, annot=True, fmt=".2f", cmap='viridis_r')
    plt.title("Mean Cost for Each Parameter Combination")
    plt.xlabel("Max Step Size")
    plt.ylabel("P Bias")
    plt.savefig('imgs/cost_heatmap.png')
    plt.close()


if __name__ == "__main__":
    compute_matrix_results()

    try:
        while True:
            compute_parameter_matrix([0.25, 0.3], [0.5, 0,8, 1, 1.2, 2.0], 5)
            compute_parameter_matrix([0.1], [0.5, 2.0], 5)
            compute_parameter_matrix([0.05], [0.8, 1.2], 5)
    except KeyboardInterrupt:
        compute_matrix_results()
    # bb, vis = init()
    # env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    # env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # planner = RRT_STAR(0.5, 2000, bb)
    # profiler = cProfile.Profile()
    # start = perf_counter()
    # profiler.enable()
    # planner.find_path(env2_start, env2_goal, "")
    # profiler.disable()
    # print(f"time taken: {perf_counter() - start}")
    # profiler.dump_stats('output.pstats')
    # print("hits:", bb.cache_hits)
    # print("misses: ", bb.cache_misses)
    # explored_configs = [np.array(vertex) for vertex in planner.tree.vertices.keys()]
    # explored_points = [bb.transform.conf2sphere_coords(conf)["wrist_3_link"][-1][:-1] for conf in explored_configs]
    # # explored spheres is explored points with a chosen radius as the 4th coord
    # explored_spheres = np.array([np.concatenate([point, [0.05]]) for point in explored_points])
    # pprint(explored_spheres)
    # vis.draw_manual_spheres(explored_spheres)
    # vis.show_conf(env2_goal)