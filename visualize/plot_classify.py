import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

def plot_loss_animation(file_path):
    # Load loss values from file
    loss_values = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                loss_values.append(float(line.strip()))
            except ValueError:
                continue  # Skip invalid lines
    
    if not loss_values:
        print("Error: No valid data found in the file.")
        return
    
    # Set Seaborn style for better visuals
    sns.set_theme(style="darkgrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, len(loss_values))
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    if min_loss == max_loss:
        min_loss -= 0.01
        max_loss += 0.01
    ax.set_ylim(min_loss, max_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Neural Network Learning")
    
    # Plot the initial empty line with a thinner line
    (line,) = ax.plot([], [], marker="o", linestyle="-", color="blue", lw=0.5)
    
    # Update function for animation
    def update(frame):
        line.set_data(range(frame), loss_values[:frame])
        return line,
    
    # Faster animation with blit=False for compatibility
    ani = animation.FuncAnimation(fig, update, frames=len(loss_values), interval=1, blit=False, repeat=False)
    
    # Show the animation
    plt.show()

# Example usage
plot_loss_animation("/home/john/playground/neuralnets/visualize/classify.log")
