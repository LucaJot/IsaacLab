import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ✅ Directory containing logged debug data
log_dir = "/home/luca/isaaclab_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5rl/logdir"

# ✅ Load all JSON files
episodes = sorted([f for f in os.listdir(log_dir) if f.endswith(".json")])
all_pos_x = []
all_pos_y = []

for episode_file in episodes:
    with open(os.path.join(log_dir, episode_file), "r") as f:
        data = json.load(f)
        all_pos_x.append(data["pos_sensor_x"])
        all_pos_y.append(data["pos_sensor_y"])

# ✅ Define compact figure size for better visibility in thesis
fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1, 1]})

# ✅ Improve font sizes for better readability in a smaller figure
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

# ✅ Plot 1: Sensor Position (X and Y over Steps) with -1 values removed while keeping step indices
for i in range(len(all_pos_x)):
    pos_x = np.array(all_pos_x[i])
    pos_y = np.array(all_pos_y[i])
    steps = np.arange(len(pos_x))

    # Filter out -1 values but keep step indices
    valid_indices = pos_x != -1
    filtered_steps = steps[valid_indices]
    filtered_x = pos_x[valid_indices]
    filtered_y = pos_y[valid_indices]

    if i == 0:  # Add labels only for the first plot to avoid duplicate legend entries
        axs[0].plot(
            filtered_steps,
            filtered_x,
            alpha=0.8,
            color="blue",
            linewidth=1,
            label="X Position",
        )
        axs[0].plot(
            filtered_steps,
            filtered_y,
            alpha=0.8,
            color="red",
            linewidth=1,
            label="Y Position",
        )
    else:
        axs[0].plot(filtered_steps, filtered_x, alpha=0.8, color="blue", linewidth=1)
        axs[0].plot(filtered_steps, filtered_y, alpha=0.8, color="red", linewidth=1)

axs[0].set_xlabel("Episode Steps")
axs[0].set_ylabel("Sensor Position")
axs[0].set_title("Sensor Position Over Time", fontsize=11)
axs[0].grid(True, linewidth=0.5)
axs[0].set_ylim(0, 1)  # ✅ Clamp sensor position between 0 and 1
axs[0].legend()

# ✅ Plot 2: Sensor Position (Y vs 1-X) with opacity fade
for i in range(len(all_pos_x)):
    pos_x = np.array(all_pos_x[i])
    pos_y = np.array(all_pos_y[i])

    valid_indices = pos_x != -1  # Remove -1 values
    filtered_x = 1 - pos_x[valid_indices]  # ✅ Apply 1 - x transformation
    filtered_y = pos_y[valid_indices]

    # Compute opacity based on step number (normalize between 0.3 and 1 for better contrast)
    opacities = np.linspace(0.3, 1.0, len(filtered_x))

    # Scatter plot with fading effect (Switched X and Y axes)
    for j in range(len(filtered_x)):
        axs[1].scatter(
            filtered_y[j], filtered_x[j], alpha=opacities[j], color="green", s=5
        )

axs[1].set_xlabel("Sensor Position Y")  # ✅ Switched labels
axs[1].set_ylabel("1 - Sensor Position X")  # ✅ Updated label
axs[1].set_title("On-Sensor Position", fontsize=11)
axs[1].grid(True, linewidth=0.5)
axs[1].set_xlim(0, 1)  # ✅ Clamp Y range to [0,1] (Now X-axis)
axs[1].set_ylim(0, 1)  # ✅ Clamp X range to [0,1] (Now Y-axis)
axs[1].set_aspect("equal")  # ✅ Ensure quadratic aspect ratio

# ✅ Reduce whitespace to optimize space in the thesis figure
plt.tight_layout(pad=1.5)

# ✅ Save plots with high DPI for better quality in thesis
plt.savefig(os.path.join(log_dir, "thesis_debug_plots.png"), dpi=300)

# ✅ Close the plot to free memory
plt.close(fig)
