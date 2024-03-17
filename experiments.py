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
from scipy.stats import norm

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
    plot_results(df_grouped, df)

def plot_results(df_grouped: pd.DataFrame, df: pd.DataFrame):
    confidence = 0.95
    # Pivot the DataFrame to create a matrix for each variable
    time_matrix = df_grouped[('time', 'mean')].unstack()
    success_matrix = df_grouped[('success', 'mean')].unstack()
    cost_matrix = df_grouped['cost'].unstack()

    # Create a heatmap for the mean time
    plt.figure(figsize=(8, 6))
    sns.heatmap(time_matrix, annot=True, fmt=".2f", cmap='RdYlBu_r')
    plt.title("Mean Time for Each Parameter Combination")
    plt.xlabel("Max Step Size")
    plt.ylabel("P Bias")
    plt.savefig('imgs/time_heatmap.png')
    plt.close()

    # Create a heatmap for the success rate
    plt.figure(figsize=(8, 6))
    sns.heatmap(success_matrix, annot=True, fmt=".2f", cmap='RdYlBu')
    plt.title("Success Rate for Each Parameter Combination")
    plt.xlabel("Max Step Size")
    plt.ylabel("P Bias")
    plt.savefig('imgs/success_rate_heatmap.png')
    plt.close()

    # Create a heatmap for the mean cost
    plt.figure(figsize=(8, 6))
    sns.heatmap(cost_matrix, annot=True, fmt=".2f", cmap='RdYlBu_r')
    plt.title("Mean Cost for Each Parameter Combination")
    plt.xlabel("Max Step Size")
    plt.ylabel("P Bias")
    plt.savefig('imgs/cost_heatmap.png')
    plt.close()

    create_errorbar_graph(df, 0.2)
    create_errorbar_graph(df, 0.05)

def create_errorbar_graph(df: pd.DataFrame, p_bias: float):
    # plot line graph for avg cost, with 95% confidence interval, for chosen p_bias
    fig, ax1 = plt.subplots(figsize=(8, 6))
    cost_df = df[df['cost'].apply(np.isfinite)]
    data = cost_df[cost_df['p_bias'] == p_bias]
    sns.lineplot(x='max_step_size', y='cost', data=data, errorbar=('ci', 95), ax=ax1, label='Cost')

    ax2 = ax1.twinx()
    data = df[df['p_bias'] == p_bias]
    sns.lineplot(x='max_step_size', y='time', data=data, errorbar=('ci', 95), ax=ax2, linestyle=':', color='orange', label='Time')
    ax1.set_title(f"Mean Cost and Time for Max Step Size, 95% confidence interval\np_bias={p_bias}")
    ax1.set_xlabel("Max Step Size")
    ax1.set_ylabel("Mean Cost")
    ax1.set_ylim(0, 10)
    ax2.set_ylabel("Mean Time")
    ax2.set_ylim(0, 10)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.savefig(f'imgs/cost_time_line_p_bias_{p_bias}.png')
    plt.close(fig)
    


if __name__ == "__main__":
    compute_matrix_results()
    try:
        while True:
            compute_parameter_matrix([0.05, 0.1, 0.2, 0.25, 0.3], [0.1, 0.3, 1.7, 2.5], 5)
            compute_parameter_matrix([0.05, 0.1, 0.2, 0.25, 0.3], [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.7, 2.0, 2.5], 5)
    except KeyboardInterrupt:
        compute_matrix_results()
