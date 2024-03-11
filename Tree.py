import numpy as np
from building_blocks import Building_Blocks


class RRTree():
    def __init__(self, root_state, bb: Building_Blocks):
        self.root_state = tuple(root_state)
        self.root_node = RRNode(root_state, 0, None)
        self.vertices = {self.root_state: self.root_node}
        self.bb = bb

    def insert_state(self, state: np.ndarray, parent_state: np.ndarray):
        cost = self.bb.edge_cost(state, parent_state)
        state = tuple(state)
        parent_state = tuple(parent_state)
        parent_node = self.vertices[parent_state]
        self.vertices[state] = RRNode(state, cost, parent_node)
    
    def distances_to_state(self, state, states):
        return [self.bb.edge_cost(state, other_state) for other_state in states]
    
    def get_nearest_state(self, state: np.ndarray):
        vertices = list(self.vertices.keys())
        distances = self.distances_to_state(state, np.array(vertices))
        nearest_vertex_index = np.argmin(distances)
        return vertices[nearest_vertex_index]

    def cost_to_state(self, state):
        return self.vertices[tuple(state)].total_cost

    def get_state_parent(self, state):
        try:
            return self.vertices[tuple(state)].parent.state
        except:
            return None
    
    def path_to_state(self, state):
        path = []
        state = tuple(state)
        if state not in self.vertices.keys():
            return np.array([]), np.inf
        cost = self.vertices[state].total_cost
        current_state = self.vertices[state]
        while current_state.parent:
            path.append(current_state.state)
            current_state = current_state.parent
        path.append(current_state.state)
        path.reverse()
        return np.array(path), cost

    def get_knn_states(self, pivot_state: np.ndarray, k: int):
        pivot_state_tuple = tuple(pivot_state)
        states_arr = list(self.vertices.keys())
        states_arr.remove(pivot_state_tuple)
        states_arr = np.array(states_arr)
        if len(states_arr) <= k:
            return states_arr

        distances = self.distances_to_state(pivot_state, states_arr)
        partitioned_states = states_arr[np.argpartition(distances, k).flatten()][:k]
        return partitioned_states

    def get_edges_as_states(self):
        return [(self.vertices[state].state, self.vertices[state].parent.state) for state in self.vertices.keys() if self.vertices[state].parent]

    def set_parent_for_state(self, state, new_parent):
        cost = self.bb.edge_cost(state, new_parent)
        state = tuple(state)
        new_parent = tuple(new_parent)
        new_parent_node = self.vertices[new_parent]
        state_node = self.vertices[state]
        state_node.parent = new_parent_node
        state_node.cost = cost
    


class RRNode():
    def __init__(self, state: tuple[float, float], cost, parent_node: "RRNode | None" = None):
        self.cost = cost
        self.parent = parent_node # type: RRNode | None
        self.state = state

    @property
    def total_cost(self):
        if self.parent:
            return self.cost + self.parent.total_cost
        return 0
