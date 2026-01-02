import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from hybrid_utils import (
    GridWorld,
    compute_future_grids,
    astar_expert,
    expert_value_labels,
    generate_cnn_sl_data,
)

# ============================================================
EXPLORATION_EPS = 0.05  # 5% of the time, take a random action instead of A*
N_FRAMES_STACKED = 4   
# ============================================================

# ============================================================
# 1. CNN framework
# ============================================================
    
class CNNHeuristic(nn.Module):
    """
    Squeeze-and-Excitation CNN.
    """
    def __init__(self, n_frames=4, n_actions=9):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(),
                nn.GroupNorm(8, cout),
                SE(cout)
            )

        self.model = nn.Sequential(
            block(n_frames, 32),
            nn.MaxPool2d(2),
            block(32, 64),
            nn.MaxPool2d(2),
            block(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r),
            nn.ReLU(),
            nn.Linear(c // r, c),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = x.mean((2,3))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ============================================================
# 2. Load pretrained CNN model
# ============================================================

def load_cnn_heuristic(model_path="cnn_heuristic.pth", size=64, device=None):
    """
    Load the pretrained CNN heuristic model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNHeuristic(size=size, n_frames=4, n_actions=9).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded CNN model from {model_path}")
    return model

# ============================================================
# 3. Generate expert training data for CNN
# ============================================================

def generate_cnn_training_data(
        n_samples=1000,
        grid_sizes=[20, 32, 48, 64],
        horizon=100,
        window_size=21,
        save_path="cnn_training_data.pkl",
        gamma=0.99
):
    """
    Generates rolling-window CNN training samples using simplified replanning.
    
    Each sample has shape (4, window_size, window_size)
      channels = [prev_window, curr_window, pos_window, goal_window]
    """

    dataset = []
    moves = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (0, 0)
    ]

    half = window_size // 2

    def extract_window(arr, r, c):
        """Extract + pad window around (r,c)."""
        H, W = arr.shape

        r0 = r - half
        r1 = r + half + 1
        c0 = c - half
        c1 = c + half + 1

        pad_top    = max(0, -r0)
        pad_bottom = max(0, r1 - H)
        pad_left   = max(0, -c0)
        pad_right  = max(0, c1 - W)

        # Extract valid region
        sub = arr[max(0, r0):min(H, r1), max(0, c0):min(W, c1)]

        # Pad with walls
        sub = np.pad(
            sub,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            constant_values=1
        )

        return sub.astype(np.float32)


    # Data generation loop
    for episode in range(n_samples):
        if (episode + 1) % 100 == 0:
            print(f"Generating episode {episode+1}/{n_samples}...")

        size = random.choice(grid_sizes)
        env = GridWorld(width=size, height=size, seed=random.randint(0, 100000))
        env.reset()

        prev_obs = env.obstacles.copy()
        cached_path = None
        t = 0
        done = False

        while not done and t < 500:

            curr_obs = env.obstacles.copy()
            (r, c) = env.pos
            (gr, gc) = env.goal

            # --------------------------------------------------
            # Simplified Replanning
            # --------------------------------------------------
            need_replan = False

            if cached_path is None or len(cached_path) < 2:
                need_replan = True
            else:
                nr, nc = cached_path[1]
                # Replan if next node is blocked now or an obstacle has moved into our current cell
                if curr_obs[nr, nc] == 1 or curr_obs[r, c] == 1:
                    need_replan = True

            if need_replan:
                future = compute_future_grids(env, horizon=horizon)
                cached_path = astar_expert(
                    future, (r, c), (gr, gc),
                    max_time=horizon
                )

            # --------------------------------------------------
            # Expert policy + value calculation
            # --------------------------------------------------
            if cached_path is None or len(cached_path) < 2:
                expert_action = random.randint(0, 8)
                values = np.zeros(9, dtype=np.float32)
                cached_path = None
            else:
                (r0, c0) = cached_path[0]
                (r1, c1) = cached_path[1]
                mv = (r1 - r0, c1 - c0)

                expert_action = moves.index(mv) if mv in moves else 8

                values = expert_value_labels(env, cached_path, gamma)

                cached_path.pop(0)    # advance along path

            # --------------------------------------------------
            # Build 21×21 window input
            # --------------------------------------------------

            prev_window = extract_window(prev_obs, r, c)
            curr_window = extract_window(curr_obs, r, c)

            # Position map
            pos_map = np.zeros((window_size, window_size), dtype=np.float32)
            pos_map[half, half] = 1.0

            # Goal map relative to window center
            goal_rel_r = gr - r + half
            goal_rel_c = gc - c + half
            goal_map = np.zeros((window_size, window_size), dtype=np.float32)
            if 0 <= goal_rel_r < window_size and 0 <= goal_rel_c < window_size:
                goal_map[goal_rel_r, goal_rel_c] = 1.0
            else:
                # Project goal to nearest edge
                goal_map[
                    np.clip(goal_rel_r, 0, window_size - 1),
                    np.clip(goal_rel_c, 0, window_size - 1)
                ] = 1.0

            frame = np.stack([
                prev_window,
                curr_window,
                pos_map,
                goal_map
            ], axis=0).astype(np.float32)

            dataset.append((frame, expert_action, values))

            # --------------------------------------------------
            # Step environment using expert action
            # --------------------------------------------------
            _, _, done, _, _ = env.step(expert_action)

            prev_obs = curr_obs
            t += 1

    # ---------------------------------------------------------
    # Save Dataset
    # ---------------------------------------------------------
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(dataset)} samples to {save_path}")
    return dataset


# ============================================================
# 3.1. Load/generate CNN training data helper
# ============================================================

def cnn_training_data_loop(
    n_samples=1000, grid_sizes=[20, 32, 48, 64], save_path="cnn_training_data.pkl"
):
    """
    Load training data if it exists, otherwise generate.
    """
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded training data from {save_path} (N={len(dataset)})")
    else:
        dataset = generate_cnn_training_data(n_samples, grid_sizes=grid_sizes, save_path=save_path)

    return dataset


# ============================================================
# 4. Train CNN
# ============================================================

def train_cnn_heuristic(
    model,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    grid_sizes=[20, 32, 48, 64],
    dataset = None,
    device=None,
    data_path="cnn_training_data.pkl",
):
    """
    Train CNN to predict value of moves.
    Loss = MSE(pred_val, true_val)
    """

    frames, actions, qvals = zip(*dataset)

    frames = np.stack(frames)
    qvals = np.stack(qvals)

    frames = torch.tensor(frames, dtype=torch.float32)
    qvals = torch.tensor(qvals, dtype=torch.float32)

    # sanity check
    assert frames.ndim == 4, f"Expected (N,C,H,W) got {frames.shape}"
    assert qvals.ndim == 2 and qvals.shape[1] == 9

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, qvals = frames.to(device), qvals.to(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    N = frames.shape[0]

    # Training
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_frames = frames[idx]
            batch_qvals = qvals[idx]

            optimizer.zero_grad()

            pred_q = model(batch_frames)

            loss = mse(pred_q, batch_qvals)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_frames.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss / N:.6f}")

    torch.save(model.state_dict(), "cnn_policy_value.pth")
    print("Saved to cnn_policy_value.pth")

    return model

def load_expert_data(path="cnn_training_data.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError("Expert dataset not found. Run expert generator first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded expert dataset: {len(data)} samples")
    return data

def cnn_sl_data_loop(
    model,
    n_episodes=200,
    grid_sizes=[20, 32, 48, 64],
    save_path="cnn_sl_training_data.pkl",
    cycle=0
):
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded RL dataset from {save_path} (N={len(dataset)})")
    else:
        dataset = generate_cnn_sl_data(
            model,
            n_episodes=n_episodes,
            grid_sizes=grid_sizes,
            save_path=save_path,
            cycle=cycle
        )
    return dataset

def train_cnn_sl(
    model,
    epochs=5,
    cycles=10,
    n_episodes=200,
    batch_size=64,
    lr=1e-3,
    grid_sizes=[20, 32, 48, 64],
    device=None,
    expert_data_path="cnn_training_data.pkl",
    buffer_size=25000  
):
    """
    Iterative SL training with a fixed-size Replay Buffer.
    - Starts with expert data.
    - Old data is automatically discarded when buffer is full.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Replay Buffer with a maxlen to automatically handle "sliding window" of data
    replay_buffer = deque(maxlen=buffer_size)

    # Load expert data to jumpstart the buffer
    expert_data = load_expert_data(expert_data_path)
    replay_buffer.extend(expert_data)
    print(f"Initial Buffer: Loaded {len(expert_data)} expert samples.")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    # ----------------------
    # Begin SL training cycles
    # ----------------------
    for cycle in range(cycles):

        print(f" SL TRAINING CYCLE {cycle+1}/{cycles}")

        # keep the file generation for diagnostics
        dataset_path = f"cnn_sl_training_data_{cycle}.pkl"
        
        new_sl_data = cnn_sl_data_loop(
            model=model,
            n_episodes=n_episodes,
            grid_sizes=grid_sizes,
            save_path=dataset_path,
            cycle=cycle
        )

        replay_buffer.extend(new_sl_data)
        print(f"Buffer Update: Added {len(new_sl_data)} samples. Total Buffer Size: {len(replay_buffer)}")

        dataset_list = list(replay_buffer)
        
        # Unzip tuples
        frames_list, actions_list, qvals_list = zip(*dataset_list)
        
        # Stack into arrays
        frames_np = np.stack(frames_list)
        qvals_np = np.stack(qvals_list)
        
        N = len(frames_np)
        indices = np.arange(N)

        # Model training
        model.train()
        
        for epoch in range(epochs):
            np.random.shuffle(indices) # Shuffle indices for this epoch
            total_loss = 0.0

            for i in range(0, N, batch_size):
                # Get batch indices
                batch_idx = indices[i : i + batch_size]
                
                # Slice data
                b_frames = frames_np[batch_idx]
                b_qvals = qvals_np[batch_idx]

                # Convert to Tensor
                b_frames_t = torch.tensor(b_frames, dtype=torch.float32, device=device)
                b_qvals_t = torch.tensor(b_qvals, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                pred_q = model(b_frames_t)
                loss = mse(pred_q, b_qvals_t)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_idx)

            avg_loss = total_loss / N
            print(f"[Cycle {cycle+1}] Epoch {epoch+1}/{epochs} — Loss = {avg_loss:.6f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"cnn_policy_value_accum_cycle{cycle}.pth")
        print(f"Saved model after cycle {cycle+1}")

    # End cycles
    torch.save(model.state_dict(), "cnn_policy_value_accum_final.pth")
    print("\n Final model: cnn_policy_value_accum_final.pth\n")

    return model
