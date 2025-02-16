import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

# Load loss values from file
file_path = "/home/john/playground/neuralnets/visualize/loss_function.log"  # Update with your file path
loss_values = pd.read_csv(file_path, header=None, names=["Loss"])

# Ask for the starting epoch
start_epoch = int(input("Enter the starting epoch (1-200): "))
epochs = np.arange(start_epoch, len(loss_values), 200)  # Select every 200th epoch
loss_subset = loss_values["Loss"].iloc[epochs]

# Set Seaborn style for better visuals
sns.set_theme(style="darkgrid")

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(start_epoch, len(loss_values))
min_loss = loss_subset.min()
max_loss = loss_subset.max()
if min_loss == max_loss:
    min_loss -= 0.01
    max_loss += 0.01
ax.set_ylim(min_loss, max_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Neural Network Learning")

# Plot the initial empty line with a thinner line
(line,) = ax.plot([], [], marker="o", linestyle="-", color="blue", lw=0.5)  # lw=0.5 makes the line thinner

# Update function for animation
def update(frame):
    line.set_data(epochs[:frame], loss_subset[:frame])  # Update only X and Y data
    return line,

# Faster animation with blit=True
ani = animation.FuncAnimation(fig, update, frames=len(epochs), interval=1, blit=True, repeat=False)

# Show the animation
plt.show()  # This will display the figure interactively
