import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Directory containing logged debug data
log_dir = "/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir"

# Load all JSON files
episodes = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
all_pos_x = []
all_pos_y = []
all_distances = []

for episode_file in episodes:
    with open(os.path.join(log_dir, episode_file), "r") as f:
        data = json.load(f)
        all_pos_x.append(data["pos_sensor_x"])
        all_pos_y.append(data["pos_sensor_y"])
        all_distances.append(data["dist_cube_cam"])

# --- Figure and GridSpec Setup ---
# 1) Use constrained_layout to align axes nicely.
fig1 = plt.figure(figsize=(6.5, 6), constrained_layout=True)
gs = fig1.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# Top-left subplot
ax1 = fig1.add_subplot(gs[0, 0])
# Top-right subplot
ax2 = fig1.add_subplot(gs[0, 1])
# Bottom subplot (spanning both columns)
ax3 = fig1.add_subplot(gs[1, :])

# --- 1) PLOT: Sensor Position Over Time ---
for i in range(len(all_pos_x)):
    pos_x = np.array(all_pos_x[i])
    pos_y = np.array(all_pos_y[i])
    steps = np.arange(len(pos_x))

    valid_indices = pos_x != -1
    filtered_steps = steps[valid_indices]
    filtered_x = pos_x[valid_indices]
    filtered_y = pos_y[valid_indices]

    if i == 0:
        # Plot with labels for legend extraction
        ax1.plot(
            filtered_steps,
            filtered_x,
            alpha=0.8,
            color="blue",
            linewidth=1,
            label="X Position",
        )
        ax1.plot(
            filtered_steps,
            filtered_y,
            alpha=0.8,
            color="red",
            linewidth=1,
            label="Y Position",
        )
    else:
        ax1.plot(filtered_steps, filtered_x, alpha=0.8, color="blue", linewidth=1)
        ax1.plot(filtered_steps, filtered_y, alpha=0.8, color="red", linewidth=1)

ax1.set_xlabel("Episode Steps")
ax1.set_ylabel("Sensor Position")
ax1.set_title("Sensor Position Over Time", fontsize=11)
ax1.grid(True, linewidth=0.5)
ax1.set_ylim(0, 1)
ax1.legend()

# --- 2) PLOT: On-Sensor Position (Scatter) ---
for i in range(len(all_pos_x)):
    pos_x = np.array(all_pos_x[i])
    pos_y = np.array(all_pos_y[i])

    valid_indices = pos_x != -1
    filtered_x = 1 - pos_x[valid_indices]
    filtered_y = pos_y[valid_indices]

    # Fading effect
    opacities = np.linspace(0.3, 1.0, len(filtered_x))

    for j in range(len(filtered_x)):
        ax2.scatter(
            filtered_y[j], filtered_x[j], alpha=opacities[j], color="green", s=5
        )

ax2.set_xlabel("Sensor Position Y")
ax2.set_ylabel("1 - Sensor Position X")
ax2.set_title("On-Sensor Position", fontsize=11)
ax2.grid(True, linewidth=0.5)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Remove or comment out the next line so it won't force a square shape:
# ax2.set_aspect("equal")

# --- 3) PLOT: Cube Distance Over Time (bottom) ---
for i in range(len(all_distances)):
    distances = np.array(all_distances[i])
    steps = np.arange(len(distances))

    valid_indices = distances != -1
    filtered_steps = steps[valid_indices]
    filtered_distances = distances[valid_indices]

    ax3.plot(
        filtered_steps,
        np.clip(filtered_distances, 0, None),
        alpha=0.8,
        color="green",
        linewidth=1,
    )

ax3.set_xlabel("Episode Steps")
ax3.set_ylabel("Cube Distance to Camera [m]")
ax3.set_title("Cube Distance Over Time", fontsize=11)
ax3.grid(True, linewidth=0.5)
ax3.set_ylim(0, None)

# --- Figure-level Legend ---
# Grab the handles/labels from ax1 (the only place we set label=...)
# handles, labels = ax1.get_legend_handles_labels()
# fig1.legend()

# Save and close
save_path_pos_dist = os.path.join(log_dir, "pos_and_dist.png")
plt.savefig(save_path_pos_dist, dpi=300)
plt.close(fig1)

save_path_pos_dist
