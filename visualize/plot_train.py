import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

# Get current directory
current_path = os.getcwd()
file_path = os.path.join(current_path, "visualize", "loss_function.log")
training_file_path = os.path.join(current_path, "training", "training.csv")  # Path to training.csv

loss_values = pd.read_csv(file_path, header=None, names=["Loss"])

try:
    with open(training_file_path, 'r') as f:
        num_training_lines = sum(1 for line in f)
except FileNotFoundError:
    print(f"Error: Training file not found at {training_file_path}")
    exit()

start_epoch = int(input(f"Enter the starting epoch (1-{num_training_lines}): "))

epochs = np.arange(start_epoch, len(loss_values), num_training_lines) 
epochs = epochs[epochs < len(loss_values)]
loss_subset = loss_values["Loss"].iloc[epochs]

# Set plot style
sns.set_theme(style="darkgrid")

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(start_epoch, len(loss_values))

# Dynamic y-axis limits
min_loss = loss_subset.min()
max_loss = loss_subset.max()

# Handle the case where min_loss == max_loss to avoid a flat line
if min_loss == max_loss:
    range_adjust = max(0.01, abs(min_loss * 0.01)) # Adjust range based on loss value
    min_loss -= range_adjust
    max_loss += range_adjust

ax.set_ylim(min_loss, max_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Neural Network Learning")

# Create initial empty line
(line,) = ax.plot([], [], linestyle="-", color="blue", lw=0.5)

# Animation update function
def update(frame):
    line.set_data(epochs[:frame + 1], loss_subset[:frame + 1])
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(epochs), interval=0, repeat=False)

# Show plot
plt.show()