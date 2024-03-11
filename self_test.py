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

if __name__ == "__main__":
    bb, vis = init()
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    planner = RRT_STAR(0.5, 2000, bb)
    profiler = cProfile.Profile()
    start = perf_counter()
    profiler.enable()
    planner.find_path(env2_start, env2_goal, "")
    profiler.disable()
    print(f"time taken: {perf_counter() - start}")
    profiler.dump_stats('output.pstats')
    print("hits:", bb.cache_hits)
    print("misses: ", bb.cache_misses)
    # explored_configs = [np.array(vertex) for vertex in planner.tree.vertices.keys()]
    # explored_points = [bb.transform.conf2sphere_coords(conf)["wrist_3_link"][-1][:-1] for conf in explored_configs]
    # # explored spheres is explored points with a chosen radius as the 4th coord
    # explored_spheres = np.array([np.concatenate([point, [0.05]]) for point in explored_points])
    # pprint(explored_spheres)
    # vis.draw_manual_spheres(explored_spheres)
    # vis.show_conf(env2_goal)