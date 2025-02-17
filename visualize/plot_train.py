import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

# Obter diretório atual
current_path = os.getcwd()
file_path = os.path.join(current_path, "visualize", "loss_function.log")

# Carregar valores da função de perda
loss_values = pd.read_csv(file_path, header=None, names=["Loss"])

# Perguntar ao usuário o epoch inicial
start_epoch = int(input(f"Enter the starting epoch (1-200): "))

# Selecionar epochs a cada 200 iterações
epochs = np.arange(start_epoch, len(loss_values), 200)
epochs = epochs[epochs < len(loss_values)]  # Garante que não saia do intervalo
loss_subset = loss_values["Loss"].iloc[epochs]

# Configurar estilo do gráfico
sns.set_theme(style="darkgrid")

# Criar figura e eixo
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

# Criar linha inicial vazia
(line,) = ax.plot([], [], linestyle="-", color="blue", lw=0.5)

# Função de atualização da animação
def update(frame):
    line.set_data(epochs[:frame + 1], loss_subset[:frame + 1])
    return line,

# Criar animação
ani = animation.FuncAnimation(fig, update, frames=len(epochs), interval=50, repeat=False)

# Mostrar gráfico
plt.show()
