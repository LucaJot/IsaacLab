import numpy as np
import matplotlib.pyplot as plt
import os


# ✅ Define the function f(x) = α * e^(-k * x)
def decay_function(x, alpha=1.0, k=0.8):
    return alpha * np.exp(-k * x)


# ✅ Generate x values
x_values = np.linspace(0, 8, 100)  # Range from 0 to 5
y_values = decay_function(x_values, alpha=1.0, k=0.8)

# ✅ Define compact figure size for better readability in a thesis
fig, ax = plt.subplots(figsize=(6, 4))

# ✅ Improve font sizes for better readability in a smaller figure
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

# ✅ Plot the function
ax.plot(x_values, y_values, color="black", linewidth=1.5)

# ✅ Labels and title
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(r"$f(x) = \alpha e^{-0.8x}, \quad \alpha = 1$", fontsize=11)

# ✅ Adjust axis limits
ax.set_xlim(0, 8)  # X-axis from 0 to 5
ax.set_ylim(0, 1.2)  # Y-axis from 0 to 0.025


# ✅ Grid for better readability
ax.grid(True, linestyle="--", linewidth=0.5)

# ✅ Reduce whitespace to optimize space in the thesis figure
plt.tight_layout(pad=1.5)

# ✅ Define save directory
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

# ✅ Save plot with high DPI for thesis use
plt.savefig(os.path.join(save_dir, "thesis_decay_plot.png"), dpi=300)

# ✅ Close the plot to free memory
plt.close(fig)
