import time
import random
import copy
from heapq import heappush, heappop
import pickle
import pandas as pd
import numpy as np
import torch
import torch
import pickle
import random
from Code.GridEnvironment import GridEnv, sample_free_cell

# ============================================================
# Util parameters:
# ============================================================

# 9 possible moves: 8 directions + stay
MOVES = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0)
]

# Penalty values for Q-function
COLLISION_PENALTY = -1.0
OBSTACLE_PROXIMITY_PENALTY_SCALE = 0.025  # Per adjacent obstacle
PROXIMITY_PENALTY_INFERENCE_SCALE = 0.5   # Scale for inference-time penalty

# Heuristic constants
WAIT_VALUE = 0.1
TOWARDS_GOAL_VALUE = 0.8
NEUTRAL_VALUE = 0.25
AWAY_FROM_GOAL_VALUE = 0.05
MIN_Q_VALUE = 0.01


# ============================================================
# 1. Wrap GridEnv from grid_environment.py to adapt for my functions
# ============================================================

class GridWorld:
    """
    A wrapper around GridEnv to provide the required outputs.
    
    Provides a 4-channel observation:
        0: obstacles at t
        1: obstacles at t-1
        2: agent position
        3: goal position
    """

    def __init__(self, width=64, height=64, dynamic=True, seed=None):
        """
        Initialize the GridWorld environment.
        
        Args:
            size: Grid size (width and height)
            n_objects: Number of moving obstacles (not directly used, GridEnv auto-calculates)
            dynamic: Whether obstacles move dynamically
            move_prob: Movement probability (not directly used by GridEnv)
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self._seed = seed
        
        # Internal GridEnv instance
        self._env = GridEnv(width=width, height=height, seed=seed)
        
        # State variables
        self.obstacles = None          # Current obstacle grid
        self.prev_obstacles = None     # Previous-step obstacle grid
        self.start = None
        self.goal = None
        self.pos = None        
        self.reset()

    def copy(self):
        """Create a deep copy of the environment."""
        return copy.deepcopy(self)

    def reset(self):
        """Return initial observation under GridEnv."""
        # Reset the underlying GridEnv
        #self._env.reset() #Must be turned off to allow consistent start/goal positions across seeds
        
        # Initialize obstacle grids
        self.obstacles = self._env.get_obstacle_grid().astype(np.int8)
        self.prev_obstacles = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Choose start position (free cell)
        start_x, start_y = sample_free_cell(self._env)
        self.start = (start_y, start_x)  # Convert to (row, col)
        
        # Choose goal position (different from start, in a free cell)
        while True:
            goal_x, goal_y = sample_free_cell(self._env)
            goal_pos = (goal_y, goal_x)  # Convert to (row, col)
            if goal_pos != self.start:
                self.goal = goal_pos
                break
        
        # Set agent position to start
        self.pos = self.start
        
        # Clear start and goal cells from obstacles
        self.obstacles[self.start[0], self.start[1]] = 0
        self.obstacles[self.goal[0], self.goal[1]] = 0
        
        return self._get_obs()

    def _update_dynamic_obstacles(self):
        """Update dynamic obstacles using GridEnv."""
        if not self.dynamic:
            return
        
        # Save previous grid
        self.prev_obstacles = self.obstacles.copy()
        
        # Update moving obstacles in GridEnv
        # GridEnv expects goal_pos as (x, y), our goal is (row, col) = (y, x)
        goal_x, goal_y = self.goal[1], self.goal[0]
        self._env.update_moving_obstacles((goal_x, goal_y))
        
        # Get updated obstacle grid
        self.obstacles = self._env.get_obstacle_grid().astype(np.int8)
        
        # Ensure start/goal are not overwritten
        self.obstacles[self.start[0], self.start[1]] = 0
        self.obstacles[self.goal[0], self.goal[1]] = 0

    def _get_obs(self):
        """Return 4-channel observation for training/inference."""
        obs = np.zeros((4, self.height, self.width), dtype=np.float32)
        
        # Channel 0: current obstacles
        obs[0] = self.obstacles.astype(np.float32)
        
        # Channel 1: previous obstacles
        if self.prev_obstacles is not None:
            obs[1] = self.prev_obstacles.astype(np.float32)
        
        # Channel 2: agent position
        r, c = self.pos
        obs[2, r, c] = 1.0
        
        # Channel 3: goal position
        gr, gc = self.goal
        obs[3, gr, gc] = 1.0
        
        return obs

    def step(self, action):
        """
        Agent takes an action. Dynamic obstacles move before agent.
        
        Args:
            action: Integer 0-8 representing direction
                0: up, 1: up-right, 2: right, 3: down-right,
                4: down, 5: down-left, 6: left, 7: up-left, 8: stay
        
        Returns:
            pos: New agent position
            reward: Reward signal
            done: Whether goal was reached
            fail: Whether agent was crushed/failed
            obs: 4-channel observation
        """

        # Move dynamic obstacles
        self._update_dynamic_obstacles()
        
        # Action to direction mapping (8 directions + stay)
        moves = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0)]
        
        dr, dc = moves[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc
        
        reward = -0.01
        done = False
        fail = False
        
        # --- Agent movement ---
        if 0 <= nr < self.height and 0 <= nc < self.width and self.obstacles[nr, nc] == 0:
            # Move agent
            old_dist = abs(self.goal[0] - r) + abs(self.goal[1] - c)
            new_dist = abs(self.goal[0] - nr) + abs(self.goal[1] - nc)
            self.pos = (nr, nc)
            
            if self.pos == self.goal:
                return self.pos, 1.0, True, False, self._get_obs()
            
            if new_dist < old_dist:
                reward += 0.01
        else:
            # Hit wall or obstacle
            reward -= -0.1
        
        # --- Bounce / Crush Handling ---
        r, c = self.pos

        # Check if an obstacle is NOW standing on the agent
        if self.obstacles[r, c] == 1:
            fail = True
            return self.pos, -1.0, False, True, self._get_obs()

        return self.pos, reward, done, fail, self._get_obs()

# ============================================================
# 2. Future-aware (expert) A*
# ============================================================

def compute_future_grids(env, horizon=200):
    """
    Simulate future grids for a given environment.
    """
    grids = [env.obstacles.copy()]
    temp_env = env.copy() 

    for x in range(horizon):
        temp_env._update_dynamic_obstacles()
        grids.append(temp_env.obstacles.copy())

    return grids

def astar_expert(grid_sequence, start, goal, max_time=200):
    """
    Expert A*: find a path in (r, c, t) that avoids future obstacle states.
    Returns a pure 2D path [(r,c), ...]
    """
    H, W = grid_sequence[0].shape

    def h(a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))  # Chebyshev distance

    moves = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (0, 0)  # wait
    ]

    open_set = []
    heappush(open_set, (0, (start[0], start[1], 0)))

    came = {}
    g = {(start[0], start[1], 0): 0}
    visited = set()

    T = min(max_time, len(grid_sequence)-1)

    while open_set:
        _, (r, c, t) = heappop(open_set)

        # Invalidate previous moves if obstacles would collide at next step
        if t + 1 < len(grid_sequence) and grid_sequence[t+1][r, c] == 1:
            node = (r, c, t)
            came.pop(node, None)
            g.pop(node, None)
            continue

        if (r, c) == goal:
            # reconstruct path
            path = [(r, c)]
            node = (r, c, t)
            while node in came:
                node = came[node]
                path.append((node[0], node[1]))
            return path[::-1]

        if (r, c, t) in visited:
            continue
        visited.add((r, c, t))

        if t >= T:
            continue

        next_grid = grid_sequence[t+1]

        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            nt = t+1

            if not (0 <= nr < H and 0 <= nc < W):
                continue
            # skip positions that will be occupied at the next time step
            if next_grid[nr, nc] == 1:
                continue

            node = (nr, nc, nt)
            cost = g.get((r, c, t), float('inf')) + 1

            if node not in g or cost < g[node]:
                g[node] = cost
                f = cost + h((nr, nc), goal)
                heappush(open_set, (f, node))
                came[node] = (r, c, t)

    return None

def expert_value_labels(env, spacetime_path, gamma=0.99):
    """
    Given the returned optimal future path, compute values for all actions.
    
    Value assignment:
    - Moves on expert path: gamma^remaining_steps
    - Moves into obstacles or out-of-bounds: collision penalty
    - Safe moves not on path: Small positive value based on goal distance heuristic
    - Moves near obstacles: Proximity-to-obstacle penalty
    
    """
    H, W = env.height, env.width

    values = np.zeros(9, dtype=np.float32)

    if spacetime_path is None:
        return values  # no expert knowledge

    pos_to_index = {pos: i for i, pos in enumerate(spacetime_path)}

    agent_pos = env.pos
    goal_pos = env.goal
    
    # Check if path actually reaches the goal
    path_reaches_goal = len(spacetime_path) > 0 and spacetime_path[-1] == goal_pos
    
    # If path doesn't reach goal, reduce base values to account for uncertainty
    path_success_factor = 1.0 if path_reaches_goal else 0.5
    
    # Compute current distance to goal for baseline
    current_dist = max(abs(goal_pos[0] - agent_pos[0]), abs(goal_pos[1] - agent_pos[1]))

    for i, (dr, dc) in enumerate(MOVES):
        nr, nc = agent_pos[0] + dr, agent_pos[1] + dc

        # Penalty for OOB
        if not (0 <= nr < H and 0 <= nc < W):
            values[i] = COLLISION_PENALTY
            continue
            
        # Obstacle penalty
        if env.obstacles[nr, nc] == 1:
            values[i] = COLLISION_PENALTY
            continue

        # Check next cell is on the expert path
        if (nr, nc) in pos_to_index:
            # correction for (0,0) move
            if dr == 0 and dc == 0:
                values[i] = WAIT_VALUE * path_success_factor
                continue 
            
            # On expert path - high positive value (scaled by success factor)
            remaining = len(spacetime_path) - pos_to_index[(nr, nc)] - 1
            values[i] = (gamma ** remaining) * path_success_factor
        else:
            remaining = len(spacetime_path) - i - 1
            # Not on expert path - compute a heuristic value
            new_dist = max(abs(goal_pos[0] - nr), abs(goal_pos[1] - nc))
            
            # Values are decreasing fractions of the chosen move based on towards/neutral/away
            if new_dist < current_dist:
                base_value = TOWARDS_GOAL_VALUE * gamma ** remaining * path_success_factor
            elif new_dist == current_dist:
                base_value = NEUTRAL_VALUE * gamma ** remaining * path_success_factor
            else:
                base_value = AWAY_FROM_GOAL_VALUE * gamma ** remaining * path_success_factor
            
            # Reduce value if near obstacles
            obstacle_penalty = _compute_obstacle_proximity_penalty(env.obstacles, nr, nc)
            values[i] = max(MIN_Q_VALUE, base_value - obstacle_penalty)

    return values


# ============================================================
# 3. Standard dynamic A* for comparison
# ============================================================

def astar(grid, start, goal):
    """Standard static A* search on the grid."""
    height = grid.shape[0]
    width = grid.shape[1]

    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  
    
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}
    visited = set()

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)

        for dr, dc in [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 0:
                neighbor = (nr, nc)
                tentative = g[current] + 1
                if neighbor not in g or tentative < g[neighbor]:
                    g[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal)
                    heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
    return None

def replanning_astar(env, start, goal, max_steps=1000):
    """
    Replanning A* run: replan after each dynamic obstacle update.
    Returns:
      path_history: list of positions visited (including start)
      goal_reached: bool
      snapshots: list of dicts per step with keys:
                 {'obstacles': np.array, 'agent_pos': (r,c), 'start':..., 'goal':...}
    """
    agent_pos = start
    path_history = [agent_pos]
    snapshots = []
    steps = 0
    goal_reached = False
    fail = False

    # initial 4 layer snapshot
    snapshots.append({
        "obstacles": env.obstacles.copy(),
        "agent_pos": agent_pos,
        "start": env.start,
        "goal": env.goal
    })

    while agent_pos != goal and steps < max_steps:
        grid = env.obstacles.copy()
        path = astar(grid, agent_pos, goal)
        if not path or len(path) < 2:
            # no path: wait (i.e., do nothing), but advance environment
            env._update_dynamic_obstacles()
            # agent_pos unchanged
            path_history.append(agent_pos)
            snapshots.append({
                "obstacles": env.obstacles.copy(),
                "agent_pos": agent_pos,
                "start": env.start,
                "goal": env.goal
            })
            steps += 1
            continue

        # move one step along path
        next_pos = path[1]
        dr = next_pos[0] - agent_pos[0]
        dc = next_pos[1] - agent_pos[1]
        moves = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1)]
        try:
            action = moves.index((dr, dc))
        except ValueError:
            # error handling: invalid move
            break

        new_pos, reward, done, fail, _ = env.step(action)
        # env.step internally advances obstacles 
        if fail:
            break
        agent_pos = new_pos
        path_history.append(agent_pos)
        snapshots.append({
            "obstacles": env.obstacles.copy(),
            "agent_pos": agent_pos,
            "start": env.start,
            "goal": env.goal
        })
        steps += 1

        if agent_pos == goal:
            goal_reached = True
            break

    return path_history, goal_reached, fail, snapshots


# ============================================================
# 4. CNN path planning
# ============================================================

def _compute_obstacle_proximity_penalty(obstacles, r, c):
    """
    Compute a relative value penalty for being close to obstacles.
    Returns a value between 0 and 0.2 based on nearby obstacle density.
    """
    H, W = obstacles.shape
    penalty = 0.0
    
    # Check 8 neighboring cells
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if obstacles[nr, nc] == 1:
                    penalty += OBSTACLE_PROXIMITY_PENALTY_SCALE  # 0.025 * 8 neighbors = 0.2 max penalty
    
    return penalty

def cnn_plan_path(env, model, max_steps=500, epsilon=0.05, device=None):
    """
    CNN-driven planner on given env (no reset).
        
    Returns:
       path_history, goal_reached, fail, snapshots
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    path_history = [env.pos]
    snapshots = [{
        "obstacles": env.obstacles.copy(),
        "agent_pos": env.pos,
        "start": env.start,
        "goal": env.goal
    }]

    done = False

    for step in range(max_steps):
        window_size = 21
        half = window_size // 2 

        # Current agent location
        r, c = env.pos
        gr, gc = env.goal

        H, W = env.obstacles.shape

        prev_obs = getattr(env, "prev_obstacles", env.obstacles).copy()
        curr_obs = env.obstacles.copy()

        # Compute window bounds
        r_min = r - half
        r_max = r + half + 1  

        c_min = c - half
        c_max = c + half + 1

        # Compute padding amounts
        pad_top    = max(0, -r_min)
        pad_left   = max(0, -c_min)
        pad_bottom = max(0, r_max - H)
        pad_right  = max(0, c_max - W)

        # Clip window region to valid grid
        r0 = max(0, r_min)
        r1 = min(H, r_max)
        c0 = max(0, c_min)
        c1 = min(W, c_max)

        # Extract windows
        curr_win = curr_obs[r0:r1, c0:c1]
        prev_win = prev_obs[r0:r1, c0:c1]

        # Pad to full window size
        curr_win = np.pad(curr_win,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        constant_values=1)

        prev_win = np.pad(prev_win,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        constant_values=1)

        # Build pos & goal maps 
        pos_map = np.zeros((window_size, window_size), dtype=np.float32)
        pos_map[half, half] = 1.0  # agent always centered

        goal_map = np.zeros((window_size, window_size), dtype=np.float32)

        # Goal location relative to window
        goal_r = gr - r + half
        goal_c = gc - c + half

        # If outside window, project to nearest edge
        goal_r = np.clip(goal_r, 0, window_size - 1)
        goal_c = np.clip(goal_c, 0, window_size - 1)

        goal_map[goal_r, goal_c] = 1.0

        # 4-channel input 

        frame = np.stack([prev_win, curr_win, pos_map, goal_map], axis=0)
        frame_t = torch.tensor(frame, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 4, 21, 21)
        
        with torch.no_grad():
            q_pred = model(frame_t).squeeze(0).cpu().numpy()  # Shape: (9,) for 9 actions

        # safety mask
        action_mask = np.ones(9, dtype=bool)
        for i, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            # Mark invalid actions
            if not (0 <= nr < H and 0 <= nc < W):
                action_mask[i] = False
            elif curr_obs[nr, nc] == 1:
                action_mask[i] = False
        
        # Apply obstacle proximity penalty to Q-values
        for i, (dr, dc) in enumerate(MOVES):
            if action_mask[i]:
                nr, nc = r + dr, c + dc
                proximity_penalty = _compute_obstacle_proximity_penalty(curr_obs, nr, nc)
                q_pred[i] -= proximity_penalty * PROXIMITY_PENALTY_INFERENCE_SCALE

        # pick action: greedy with epsilon and safety masking
        if random.random() < epsilon:
            # Random action, but prefer safe actions
            safe_actions = np.where(action_mask)[0]
            if len(safe_actions) > 0:
                action_idx = int(np.random.choice(safe_actions))
            else:
                action_idx = random.randint(0, 8)
        else:
            # Greedy with mask
            masked_q = q_pred.copy()
            masked_q[~action_mask] = -float('inf')
            action_idx = int(np.argmax(masked_q))

        new_pos, reward, done, fail, _ = env.step(action_idx)
        if fail:
            # cannot proceed further
            done = False
            break
        path_history.append(new_pos)
        # record snapshot after stepping
        snapshots.append({
            "obstacles": env.obstacles.copy(),
            "agent_pos": new_pos,
            "start": env.start,
            "goal": env.goal
        })
        if done:
            break

    return path_history, done, fail, snapshots


# ============================================================
# 5. Generate CNN data for self learning
# ============================================================

def generate_cnn_sl_data(
    model,
    n_episodes=200,
    grid_sizes=[20, 32, 48, 64],
    gamma=0.99,
    epsilon=0.1,
    max_steps=500,
    save_path="cnn_rl_training_data.pkl",
    device=None,
    success_only=True,
    cycle=0
):
    """
    Generate self-learning training dataset from CNN.
    Produces (frame, action, target_q) tuples (as in expert dataset).
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_samples = []
    # counters for each cycle
    completed_episodes = 0
    failed_episodes = 0
    timeout_episodes = 0

    for ep in range(n_episodes):

        if (ep + 1) % 100 == 0:
            print(f"[RL] Running episode {ep+1}/{n_episodes}")

        size = random.choice(grid_sizes)

        if cycle==29:
            df = pd.read_csv("episodes_train.csv")
            row = df.iloc[ep]
            width, height = int(row['width']), int(row['height'])
            seed = int(row['env_seed'])
            env = GridWorld(width=width, height=height, seed=seed)
        else:
            env = GridWorld(width=size, height=size, seed=random.randint(0, 100000))
        env.reset()

        trajectory = []   # list of (frame, action, q_pred, reward, was_risky)
        done = False
        fail = False

        for step in range(max_steps):

            # Build local CNN window
            window_size = 21
            half = window_size // 2     # half = 10 for 21×21 window

            # Current agent location
            r, c = env.pos
            gr, gc = env.goal

            H, W = env.obstacles.shape

            prev_obs = getattr(env, "prev_obstacles", env.obstacles).copy()
            curr_obs = env.obstacles.copy()

            # Compute window bounds
            r_min = r - half
            r_max = r + half + 1   # +1 because slice end is exclusive

            c_min = c - half
            c_max = c + half + 1

            # Compute padding amounts
            pad_top    = max(0, -r_min)
            pad_left   = max(0, -c_min)
            pad_bottom = max(0, r_max - H)
            pad_right  = max(0, c_max - W)

            # Clip window region to valid grid
            r0 = max(0, r_min)
            r1 = min(H, r_max)
            c0 = max(0, c_min)
            c1 = min(W, c_max)

            # Extract windows
            curr_win = curr_obs[r0:r1, c0:c1]
            prev_win = prev_obs[r0:r1, c0:c1]

            # Pad to full window size
            curr_win = np.pad(curr_win,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            constant_values=1)

            prev_win = np.pad(prev_win,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            constant_values=1)

            # Build pos & goal maps
            pos_map = np.zeros((window_size, window_size), dtype=np.float32)
            pos_map[half, half] = 1.0  # agent always centered

            goal_map = np.zeros((window_size, window_size), dtype=np.float32)

            # Goal location relative to window
            goal_r = gr - r + half
            goal_c = gc - c + half

            # If outside window, project to nearest edge
            goal_r = np.clip(goal_r, 0, window_size - 1)
            goal_c = np.clip(goal_c, 0, window_size - 1)

            goal_map[goal_r, goal_c] = 1.0

            # 4-channel input

            frame = np.stack([prev_win, curr_win, pos_map, goal_map], axis=0)
            frame_t = torch.tensor(frame, dtype=torch.float32, device=device).unsqueeze(0)

            # Predict Q-values
            with torch.no_grad():
                q_pred = model(frame_t).squeeze(0).cpu().numpy()

            # Compute action mask
            action_mask = np.ones(9, dtype=bool)
            for i, (dr, dc) in enumerate(MOVES):
                nr, nc = r + dr, c + dc
                # Mark invalid actions
                if not (0 <= nr < H and 0 <= nc < W):
                    action_mask[i] = False
                elif curr_obs[nr, nc] == 1:
                    action_mask[i] = False

            # ε-greedy action selection
            if random.random() < epsilon:
                # Random action, but prefer safe actions
                safe_actions = np.where(action_mask)[0]
                if len(safe_actions) > 0:
                    action = int(np.random.choice(safe_actions))
                else:
                    action = random.randint(0, 8)
            else:
                # Greedy with mask
                masked_q = q_pred.copy()
                masked_q[~action_mask] = -float('inf')
                action = int(np.argmax(masked_q))

            # Track if we took a risky action (near obstacles)
            was_risky = not action_mask[action]

            # Step environment
            pos, reward, done, fail, _ = env.step(action)

            if fail:
                failed_episodes += 1
                break
            if done:
                completed_episodes += 1
                break

            # save for later return-labeling, include curr_obs and agent_pos for proximity penalties
            agent_pos = (r, c)  # Position before the step
            trajectory.append((frame, action, q_pred, reward, was_risky, action_mask, curr_obs.copy(), agent_pos))

        # Track timeout episodes
        if not done and not fail:
            timeout_episodes += 1

        # Only use successful episodes for training (depending on success_only)
        if done and not fail:
            outcome_value = 1.0
        else:
            outcome_value = 0.0
            if success_only:
                continue  # Skip failed episodes

        # Apply obstacle proximity penalties consistent with expert data labeling
        Gt = outcome_value

        for frame, action, old_q_pred, step_reward, was_risky, action_mask, curr_obs, agent_pos in reversed(trajectory):
            
            target_q_vector = old_q_pred.copy()
            H, W = curr_obs.shape
            r, c = agent_pos
            # current Chebyshev distance
            goal_r, goal_c = env.goal
            current_dist = max(abs(goal_r - r), abs(goal_c - c))
            
            for i, (dr, dc) in enumerate(MOVES):
                nr, nc = r + dr, c + dc

                # Assign collision penalty if OOB or risky move
                if not action_mask[i] or not (0 <= nr < H and 0 <= nc < W):
                    target_q_vector[i] = COLLISION_PENALTY
                    continue

                # Proximity penalty
                proximity_penalty = _compute_obstacle_proximity_penalty(curr_obs, nr, nc)

                # Taken action: use discounted return minus proximity penalty
                if i == action:
                    target_q_vector[i] = max(MIN_Q_VALUE, Gt - proximity_penalty)
                    continue

                # Untaken safe actions: compute heuristic base based on progress toward goal
                new_dist = max(abs(goal_r - nr), abs(goal_c - nc))
                if dr == 0 and dc == 0:
                    base_value = WAIT_VALUE
                elif new_dist < current_dist:
                    base_value = TOWARDS_GOAL_VALUE
                elif new_dist == current_dist:
                    base_value = NEUTRAL_VALUE
                else:
                    base_value = AWAY_FROM_GOAL_VALUE

                # Combine model prior and heuristic: preserve prior where it is higher, but apply penalty
                heuristic_val = max(MIN_Q_VALUE, base_value * Gt - proximity_penalty)
                prior_val = max(MIN_Q_VALUE, old_q_pred[i] - proximity_penalty)
                target_q_vector[i] = max(prior_val, heuristic_val)

            all_samples.append((frame, action, target_q_vector))
            
            # Decay final outcome
            Gt = gamma * Gt

    # Compute completion rate out of all episodes
    completion_rate = completed_episodes / n_episodes
    print(f"✅ Completed: {completed_episodes}, Failed: {failed_episodes}, Timeout: {timeout_episodes}")
    print(f"✅ Completion rate: {completion_rate:.2%}")

    with open(save_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"💾 RL dataset generated: {len(all_samples)} samples → {save_path}")
    return all_samples

# ============================================================
# 6. Evaluation
# ============================================================

def evaluate(model, n_tests=30, device=None, file=None):
    """
    Evaluate paths in the same environments using A* and hybrid approach.
    """

    device = device or torch.device("cpu")
    model.to(device)
    model.eval()

    astar_success = 0
    cnn_success = 0
    astar_lengths = []
    cnn_lengths = []
    astar_times = []
    cnn_times = []
    df = pd.read_csv(file)

    for t in range(n_tests):
        row = df.iloc[t]
        width, height = int(row['width']), int(row['height'])
        seed = int(row['env_seed'])

        #  Create environment
        base_env = GridWorld(width=width, height=height, seed=seed)   
        #base_env.reset() # Do not reset to preserve seed

        # Clone environment
        env_astar = base_env.copy()
        env_cnn   = base_env.copy()

        #  A* Evaluation
        t0 = time.time()
        path_astar, reached_astar, _, _ = replanning_astar(
            env_astar, env_astar.start, env_astar.goal
        )
        t1 = time.time()

        elapsed_astar = t1 - t0
        astar_times.append(elapsed_astar)

        if reached_astar and path_astar:
            astar_success += 1
            astar_lengths.append(len(path_astar))
        else:
            astar_lengths.append(0)

        # Diagnostics
        #print(f"A* reached: {reached_astar}, length={len(path_astar)}, time={elapsed_astar:.4f}s, width={width}, height={height}, start= {env_astar.start}, end = {env_astar.goal}")

        #  CNN Evaluation
        t0 = time.time()
        path_cnn, reached_cnn, _, _ = cnn_plan_path(
            env_cnn, model, max_steps=400, epsilon=0.0, device=device
        )
        t1 = time.time()

        elapsed_cnn = t1 - t0
        cnn_times.append(elapsed_cnn)

        if reached_cnn and path_cnn:
            cnn_success += 1
            cnn_lengths.append(len(path_cnn))
        else:
            cnn_lengths.append(0)

        # Diagnostics
        #print(f"CNN reached: {reached_cnn}, length={len(path_cnn)}, time={elapsed_cnn:.4f}s")

    # Summary statistics
    print("\n================= Evaluation =================")
    print(f"A* success:    {astar_success}/{n_tests}")
    print(f"CNN success:   {cnn_success}/{n_tests}")
    print("---------------------------------------------")
    print(f"A* avg length: {np.mean(astar_lengths):.2f}")
    print(f"CNN avg length:{np.mean(cnn_lengths):.2f}")
    print("---------------------------------------------")
    print(f"A* avg time:   {np.mean(astar_times):.6f}s")
    print(f"CNN avg time:  {np.mean(cnn_times):.6f}s")
    print(f"A* 90% time:   {np.percentile(astar_times, 90):.6f}s")
    print(f"CNN 90% time:  {np.percentile(cnn_times, 90):.6f}s")
    print("================================================\n")

    # Paired comparison: where both methods produced a path
    astar_arr = np.array(astar_lengths)
    cnn_arr = np.array(cnn_lengths)
    astar_t = np.array(astar_times)
    cnn_t = np.array(cnn_times)

    paired_mask = (astar_arr > 0) & (cnn_arr > 0)
    n_paired = int(paired_mask.sum())

    if n_paired == 0:
        print("No paired cases where both A* and CNN produced a path (length > 0).")
    else:
        astar_p_len = astar_arr[paired_mask]
        cnn_p_len = cnn_arr[paired_mask]
        astar_p_time = astar_t[paired_mask]
        cnn_p_time = cnn_t[paired_mask]

        mean_astar_len = astar_p_len.mean()
        mean_cnn_len = cnn_p_len.mean()
        mean_astar_time = astar_p_time.mean()
        mean_cnn_time = cnn_p_time.mean()

        mean_len_diff = (mean_cnn_len - mean_astar_len)
        mean_time_diff = (mean_cnn_time - mean_astar_time)
        pct_len_change = (mean_len_diff / mean_astar_len * 100) if mean_astar_len != 0 else float('nan')
        pct_time_change = (mean_time_diff / mean_astar_time * 100) if mean_astar_time != 0 else float('nan')

        print("\n--- Paired comparison (both produced a path) ---")
        print(f"Paired cases:    {n_paired}/{n_tests}")
        print(f"A* mean length:  {mean_astar_len:.2f}")
        print(f"CNN mean length: {mean_cnn_len:.2f}")
        print(f"Mean length diff (CNN - A*): {mean_len_diff:.2f} ({pct_len_change:.2f}%)")
        print("---------------------------------------------")
        print(f"A* mean time:    {mean_astar_time:.6f}s")
        print(f"CNN mean time:   {mean_cnn_time:.6f}s")
        print(f"Mean time diff (CNN - A*): {mean_time_diff:.6f}s ({pct_time_change:.2f}%)")
        print("---------------------------------------------")
