import torch
from hybrid_utils import (
    evaluate,
)
from hybrid_cnn_training import (
    CNNHeuristic,
    cnn_training_data_loop,
    train_cnn_heuristic,
    train_cnn_sl,
)
import os

# ============================================================
# 1. Setup
# ============================================================
grid_sizes=[32, 56, 80, 104]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#cnn_model_path = "cnn_heuristic.pth" # or "cnn_policy_value_accum_final.pth"
cnn_model_path = "cnn_policy_value_accum_final.pth"
training_data_path = "cnn_training_data.pkl"
test_data_path = "cnn_test_data.pkl"

# ============================================================
# 2. Train CNN Model
# ============================================================
cnn_heuristic = CNNHeuristic().to(device)

if os.path.exists(cnn_model_path):
    print(f"Loading existing CNN model from {cnn_model_path}")
    cnn_heuristic.load_state_dict(torch.load(cnn_model_path))
    cnn_heuristic.eval()
else:
    print("Training CNN model:")
    data = cnn_training_data_loop(n_samples=1000, grid_sizes=grid_sizes, save_path=training_data_path)
    train_cnn_heuristic(cnn_heuristic, epochs=3, batch_size=16, lr=1e-3, grid_sizes=grid_sizes, dataset=data, device=device)
    torch.save(cnn_heuristic.state_dict(), cnn_model_path)
    print(f"CNN model saved to {cnn_model_path}")

print(f"====== SL Cycle ======")
cnn_heuristic = train_cnn_sl(cnn_heuristic, epochs=1, cycles = 30, n_episodes=500, batch_size=64, grid_sizes=grid_sizes, device=device)

# ============================================================
# 3. Evaluate A* and hybrid approach
# ============================================================
print("\nRunning evaluation:")
metrics = evaluate(cnn_heuristic, n_tests=500, device=device, file = "episodes_train.csv")
metrics = evaluate(cnn_heuristic, n_tests=3500, device=device, file = "episodes_valid.csv")
metrics = evaluate(cnn_heuristic, n_tests=10000, device=device, file = "episodes_test.csv")

# ============================================================
# 4. Visualization script
# ============================================================

env_cnn = #Add copy of environment 
cnn_model_path = "cnn_policy_value_accum_final.pth"
cnn_heuristic = CNNHeuristic().to(device)
cnn_heuristic.load_state_dict(torch.load(cnn_model_path))
cnn_heuristic.eval()
# Returns 
path_cnn, _, _, _ = cnn_plan_path(env_cnn, model, max_steps=400, epsilon=0)
