import numpy as np
from kinematics import UR5e_PARAMS as UR_PARAMS
from kinematics import Transform
from environment import Environment
from scipy.spatial.distance import cdist
from functools import lru_cache

class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''
    def __init__(self, transform: Transform, ur_params: UR_PARAMS, env: Environment, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params # type: UR_PARAMS
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3 ,0.2 ,0.1 ,0.07 ,0.05])
        self.checked_states = {} # type: dict[tuple[float, float], bool]
        self.cache_hits = 0
        self.cache_misses = 0
        self.parts = self.ur_params.ur_links
        self.arm_part_combinations = []
        for i in range(len(self.parts)):
            for j in range(i + 2, len(self.parts)):
                self.arm_part_combinations.append((self.parts[i], self.parts[j]))

        global_sphere_coords = self.transform.conf2sphere_coords([0,0,0,0,0,0])
        self.sphere_radii = np.concatenate([np.repeat(self.ur_params.sphere_radius[part], len(global_sphere_coords[part])) for part in self.parts])

        
    def sample(self, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        """
        if np.random.uniform(0, 1) < self.p_bias:
            return goal_conf
        constraints = np.array(list(self.ur_params.mechamical_limits.values()))
        conf = np.random.uniform(constraints[:, 0], constraints[:, 1])
        return np.array(conf)
        
    def is_in_collision(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration 
        """
        conf_tuple = tuple(conf)
        try:
            res = self.checked_states[conf_tuple]
            self.cache_hits += 1
            return res
        except KeyError:
            self.cache_misses += 1
            self.checked_states[conf_tuple] = True
        # hint: use self.transform.conf2sphere_coords(), self.ur_params.sphere_radius, self.env.obstacles
        # global sphere coords: {link name: list of spheres}, s.t. list of spheres = [(x, y, z, [SOMETHING??])]
        global_sphere_coords = self.transform.conf2sphere_coords(conf)
    
        # arm - floor collision
        for key, spheres in global_sphere_coords.items():
            if key == "shoulder_link":
                continue
            radius = self.ur_params.sphere_radius[key]
            if any((sphere[2] - radius < 0 or sphere[0] + radius > 0.4) for sphere in spheres):
                # print("Collision with floor detected!")
                return True
    
        # arm - obstacle collision
        if self.env.obstacles.size > 0:
            obstacle_spheres = self.env.obstacles
            robot_spheres = np.concatenate(list(global_sphere_coords.values()), axis=0)
            distances = cdist(robot_spheres[:, :-1], obstacle_spheres)
            # sphere_radii = np.concatenate([np.repeat(self.ur_params.sphere_radius[part], len(global_sphere_coords[part])) for part in parts])
            differences = distances - self.sphere_radii[:, None] - self.env.radius
            if np.any(differences < 0):
                return True
        
        # get list of combination of robot parts, ignoring parts that are adjacent as they always collide at their connection point
        # arm - arm collision
        part_spheres = {part: np.array(global_sphere_coords[part]) for part in self.parts}
        for part_1, part_2 in self.arm_part_combinations:
            distances = cdist(part_spheres[part_1][:, :-1], part_spheres[part_2][:, :-1])
            differences = distances - (self.ur_params.sphere_radius[part_1] + self.ur_params.sphere_radius[part_2])
            if np.any(differences < 0):
                return True

        self.checked_states[conf_tuple] = False
        return False
    
        
    
    def local_planner(self, prev_conf ,current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        num_configs = int(np.max(np.abs(current_conf - prev_conf)) / self.resolution) + 1
        if num_configs < 3:
            num_configs = 3
        configs = np.linspace(prev_conf, current_conf, num_configs, True)
        return not any(self.is_in_collision(config) for config in configs)

    
    def edge_cost(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1-conf2,2)) ** 0.5
