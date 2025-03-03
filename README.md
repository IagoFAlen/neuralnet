# Neural Network Classification & Visualization

> **Tags:** Neural Network, C++, Machine Learning, AI, Deep Learning, Backpropagation, OpenGL, SDL2, Visualization, Optimization, Training, Classification

This project implements an adaptive neural network with configurable learning rate, lambda and epochs. The neural network dynamically adjusts the number of layers and neurons per layer, making it suitable for various types of data and learning tasks. It also includes a visualization system using OpenGL and SDL2 to render the neural network's structure and training process.

## Features

- **Dynamic Neural Network:** Supports configurable layers and neurons.
- **Training & Prediction:** Implements feed-forward, backpropagation, and weight updates.
- **Regularization & Optimization:** Uses L2 regularization and gradient clipping.
- **Visualization:** Renders the neural network structure and tracks loss function progression.
- **Command-line Arguments:** Allows configuring learning rate, epochs, lambda regularization, and rendering mode.
- **File Persistence:** Supports saving and loading trained models.

## Installation

### Requirements

Ensure you have the following dependencies installed:

- [LIST Library](https://github.com/IagoFAlen/list) (for linked list management)
- `g++` (or any compatible C++ compiler)
- `CMake`
- `SDL2`
- `GLFW` & `OpenGL`
- `pthread`

### Build Instructions

```sh
mkdir build && cd build
cmake ..
make
```

## Usage

Run the program with command-line arguments:

```sh
./path/to/neuralnet -l <num_layers> -n <neurons_per_layer> [--lr <learning_rate>] [--lambda <lambda>] [--epochs <epochs>] [--render]
```

### Example:

```sh
./out/build/neuralnets -l 5 -n 7 20 20 20 10 --lr 0.01 --lambda 0.001 --epochs 1000 --render
```

This initializes a neural network with 3 layers, neuron sizes `[7, 20, 20, 20, 10]`, a learning rate of 0.01, lambda of 0.001, and 1000 epochs with rendering enabled.

## File Persistence

### Save Model:
The program saves neural network state after training process ends.

### Load Model:
Once the neural network was trained, it can be loaded with previous values.
```sh
./neural_net --load <file_path>
```

## Visualization

If `--render` is enabled, the OpenGL-based visualization system will display:

- The neural network's structure.
- Activation values of neurons.
- A real-time loss function graph.

## Example Video

```md
[![Neural Network Demo](https://img.youtube.com/vi/ZzN9SLgNDDk/0.jpg)](https://www.youtube.com/watch?v=ZzN9SLgNDDk)
```

## Contributing

If you want to contribute, feel free to submit a pull request or report issues.

## License

This project is licensed under the MIT License.
