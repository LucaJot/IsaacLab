import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ✅ Directory containing logged debug data
log_dir = "/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir"

# ✅ Load all JSON files
episodes = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
all_distances = []
all_mean_torque = []

for episode_file in episodes:
    with open(os.path.join(log_dir, episode_file), "r") as f:
        data = json.load(f)
        all_distances.append(data["dist_cube_cam"])
        all_mean_torque.append(data["mean_torque"])

# ✅ Create a figure for distance and mean torque visualization with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [1, 1]})

# ✅ Plot 1: Cube Distance Over Time
for i in range(len(all_distances)):
    steps = np.arange(len(all_distances[i]))
    axs[0].plot(
        steps, np.clip(all_distances[i], 0, None), alpha=0.8, color="green", linewidth=1
    )

axs[0].set_xlabel("Episode Steps")
axs[0].set_ylabel("Cube Distance to Camera")
axs[0].set_title("Cube Distance Over Time", fontsize=11)
axs[0].grid(True, linestyle="--", linewidth=0.5)
axs[0].set_ylim(0, None)

# ✅ Plot 2: Mean Torque Over Time (Clipped to ±200 N)
for i in range(len(all_mean_torque)):
    steps = np.arange(len(all_mean_torque[i]))
    axs[1].plot(
        steps, np.clip(all_mean_torque[i], -400, 400), color="purple", linewidth=1
    )

axs[1].set_xlabel("Episode Steps")
axs[1].set_ylabel("Mean Torque (N)")
axs[1].set_title("Mean Torque Over Time", fontsize=11)
axs[1].grid(True, linestyle="--", linewidth=0.5)
axs[1].set_ylim(0, 200)  # ✅ Clipping the torque values

# ✅ Reduce whitespace to optimize space in the thesis figure
plt.tight_layout(pad=1.5)

# ✅ Save the distance and torque plot as a separate file
save_path_distance_torque = os.path.join(log_dir, "distance_torque.png")
plt.savefig(save_path_distance_torque, dpi=300)

# ✅ Close the plot to free memory
plt.close(fig)

# ✅ Return the path where the figure is saved
save_path_distance_torque
