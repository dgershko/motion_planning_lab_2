import numpy as np
import time
from RRTTree import RRTTree
from Tree import RRTree
from building_blocks import Building_Blocks
from pprint import pprint

class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb: Building_Blocks):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb

    
    def find_path(self, start_conf, goal_conf, filename, return_cost=False):
        """Implement RRT-STAR - Return a path as numpy array"""
        self.tree = RRTree(start_conf, self.bb)
        
        i = 1

        while i < self.max_itr:
            rand_state = self.bb.sample(goal_conf)
            nearest_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(nearest_state, rand_state)
            if self.bb.local_planner(new_state, nearest_state):
                self.tree.insert_state(new_state, nearest_state)

                # the * part of the algorithm
                current_k = self.get_k_num(i)
                near_states = self.tree.get_knn_states(new_state, current_k)
                # filter out states with illegal edges
                near_states = [state for state in near_states if self.bb.local_planner(state, new_state)]
                for state in near_states:
                    self.rewire(state, new_state)
                for state in near_states:
                    self.rewire(new_state, state)

                if np.array_equal(new_state, goal_conf):
                    print("found goal state!")
                    break
            i += 1
        
        end_time = time.perf_counter()
        # print(f"algorithm finished in {end_time - time_start} seconds")
        # print(f"added {len(self.tree.vertices.keys())} configurations to tree")
        path, cost = self.tree.path_to_state(goal_conf)
        # print(f"found path with cost {cost}:")
        # pprint(path)
        if return_cost:
            return path, cost
        return path


    def extend(self, x_near, x_random)-> np.array:
        '''
        Implement the Extend method
        @param x_near - Nearest Neighbor
        @param x_random - random sampled configuration
        return the extended configuration
        '''
        direction = x_random - x_near
        distance = np.linalg.norm(direction)
        if distance <= self.max_step_size:
            return x_random
        unit_direction_vector = direction / distance
        return x_near + unit_direction_vector * self.max_step_size
       
    
    def rewire(self, x_potential_parent, x_child) -> None:
        '''
        Implement the rewire method
        @param x_potential_parent_id - candidte to become a parent
        @param x_child_id - the id of the child vertex
        return None
        '''
        # if self.bb.ed(potential_parent, child):
        cost = self.bb.edge_cost(x_potential_parent, x_child)
        if self.tree.cost_to_state(x_potential_parent) + cost < self.tree.cost_to_state(x_child):
            self.tree.set_parent_for_state(state=x_child, new_parent=x_potential_parent)


    def get_shortest_path(self, dest):
        '''
        Returns the path and cost from some vertex to Tree's root
        @param dest - the id of some vertex
        return the shortest path and the cost
        '''
        return self.tree.path_to_state(dest)
    
    def get_k_num(self, i):
        '''
        Determines the number of K nearest neighbors for each iteration
        '''
        if i < 300:
            k_num = 1
        elif 300 <= i < 600:
            k_num = 3
        elif 600 <= i < 1000:
            k_num=5
        elif 1000 <= i < 1500:
            k_num=6
        else:
            k_num = 7
        return k_num
