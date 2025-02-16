import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

def plot_loss_animation(file_name):
    current_path = os.getcwd()
    file_path = os.path.join(current_path, "visualize", file_name)

    loss_values = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                try:
                    loss_values.append(float(line.strip()))
                except ValueError:
                    continue  # Ignora linhas inválidas
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        return

    if not loss_values:
        print("Erro: Nenhum dado válido encontrado no arquivo.")
        return

    sns.set_theme(style="darkgrid")

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

    (line,) = ax.plot([], [], marker="o", linestyle="-", color="blue", lw=0.5)

    def update(frame):
        line.set_data(range(frame), loss_values[:frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(loss_values), interval=50, blit=False, repeat=False)

    plt.show()

plot_loss_animation("classify.log")
