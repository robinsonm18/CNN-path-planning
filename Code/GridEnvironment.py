import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from matplotlib import pyplot as plt
import pandas as pd

# 8 moves (possible actions)
# grid(y,x), +y -> down, +x -> right
DIRS = [
    (1, 1),    # 0 - down-right
    (1, 0),    # 1 - down
    (1, -1),   # 2 - down-left
    (0, -1),   # 3 - left
    (-1, -1),  # 4 - up-left
    (-1, 0),   # 5 - up
    (-1, 1),   # 6 - up-right
    (0, 1)     # 7 - right
]
DIR_VECS = np.array(DIRS, dtype=int)

class MovingObstacle:
    def __init__(self, pos, size, grid_w, grid_h, rng):
        self.pos = np.array(pos, dtype=int)
        self.size = size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.prev_dir = None
        self.rng = rng

    def step(self, occ_grid, goal_pos):
        directions = np.arange(8)

        # Compute probabilities for each direction
        if self.prev_dir is None:
            probs = np.ones(8)/8 # initially uniform probabilities
        else:
            # 0.7 weight to prev_dir, 0.1 weight to adj dirs, and 0.02 to others
            probs = np.ones(8)/50
            probs[self.prev_dir] = 0.7
            probs[(self.prev_dir+1)%8] = 0.1
            probs[(self.prev_dir-1)%8] = 0.1
            probs /= probs.sum() # normalize in case probability weights are changed later

        # Sample direction
        choice = self.rng.choice(directions, p=probs)
        move = DIR_VECS[choice]
        new_pos = self.pos + move
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_w - self.size[0])
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_h - self.size[1])

        region = occ_grid[new_pos[1]:new_pos[1]+self.size[1], new_pos[0]:new_pos[0]+self.size[0]]
        overlaps_goal = (goal_pos[0] >= new_pos[0] and goal_pos[0] < new_pos[0]+self.size[0] and
                         goal_pos[1] >= new_pos[1] and goal_pos[1] < new_pos[1]+self.size[1])

        # If region occupied (by static or moving), reverse direction
        if np.all(region==0) and not overlaps_goal:
            self.pos = new_pos
            self.prev_dir = choice
        else:
            # Bounce: reverse previous direction
            if self.prev_dir is not None:
                self.prev_dir = (self.prev_dir + 4) % 8

# Environment setup
# Option to include seed to test/observe agent in the same environment
class GridEnv:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)
        self.n_moving_obstacles = self.rng.randint(int(min(width,height)/6), int(min(width,height)/3))
        self.max_obs_size = int(min(width,height)/8)
        self.reset()

    # Random connected filled shape generator for static obstacles
    def generate_filled_shape(self, max_cells):
        shape_mask = np.zeros((self.height, self.width), dtype=int)
        start_y = self.rng.randint(0, self.height)
        start_x = self.rng.randint(0, self.width)
        shape_cells = [(start_y, start_x)]
        shape_mask[start_y, start_x] = 1

        while len(shape_cells) < max_cells:
            # pick a random cell from current shape
            idx = self.rng.randint(len(shape_cells))
            y, x = shape_cells[idx]

            # valid 4 neighbors
            neighbors = [(y+dy, x+dx) for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= y+dy < self.height and 0 <= x+dx < self.width and shape_mask[y+dy, x+dx]==0]
            if not neighbors:
                break
            ny, nx = neighbors[self.rng.randint(len(neighbors))]
            shape_cells.append((ny, nx))
            shape_mask[ny, nx] = 1

        return shape_mask

    def reset(self):
        self.static_obstacles = self.generate_static_obstacles()
        self.moving_obstacles = self.generate_moving_obstacles()

    def generate_static_obstacles(self): # generates static obstacles
        grid = np.zeros((self.height, self.width), dtype=np.float32)
        n_obs = self.rng.randint(int(min(self.height,self.width)/20), int(min(self.height,self.width)/8))
        for _ in range(n_obs):
            placed = False
            while not placed:
                max_cells = self.rng.randint(4, self.max_obs_size**2)
                shape_mask = self.generate_filled_shape(max_cells)
                if np.all((grid + shape_mask) <= 1):
                    grid = np.maximum(grid, shape_mask)
                    placed = True
        return grid

    def generate_moving_obstacles(self): # initializes moving obstacles
        mobs = []
        occ_grid = deepcopy(self.static_obstacles) # temporary grid to avoid conflicts with static obstacles
        for _ in range(self.n_moving_obstacles):
            placed = False
            while not placed:
                w, h = self.rng.randint(2, self.max_obs_size,2)
                x = self.rng.randint(0, self.width - w)
                y = self.rng.randint(0, self.height - h)
                region = occ_grid[y:y+h, x:x+w]
                if np.all(region==0):
                    mob = MovingObstacle((x,y), (w,h), self.width, self.height, self.rng)
                    mobs.append(mob)
                    occ_grid[y:y+h, x:x+w] = 1.0
                    placed = True
        return mobs

    def update_moving_obstacles(self, goal_pos):
        # Get all obstacles
        occ_grid = self.get_obstacle_grid()

        # Update moving obstacles in random order so the same mob doesn't always get priority
        order = list(range(len(self.moving_obstacles)))
        self.rng.shuffle(order)

        for i in order:
            mob = self.moving_obstacles[i]
            x, y = mob.pos
            w, h = mob.size

            occ_grid[y:y+h, x:x+w] = 0

            # Step while other mobs remain in occ_grid
            mob.step(occ_grid, goal_pos)
            x2, y2 = mob.pos
            occ_grid[y2:y2+h, x2:x2+w] = 1.0


    def get_obstacle_grid(self): # combines both static and current moving obstacles into one universal grid
        grid = deepcopy(self.static_obstacles)
        for mob in self.moving_obstacles:
            x,y = mob.pos
            w,h = mob.size
            grid[y:y+h, x:x+w] = 1.0
        return grid
    
# Sampling
def sample_free_cell(env):
    occ = env.get_obstacle_grid()
    while True:
        x = env.rng.randint(0, env.width)
        y = env.rng.randint(0, env.height)
        if env.static_obstacles[y,x]==0 and occ[y,x]==0: return (x,y)



