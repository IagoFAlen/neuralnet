#pragma once

#include "neuralnetwork.hpp"
#include <GLFW/glfw3.h>

using namespace neuralnets;

namespace render {
    void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void init_opengl();
    void render_network(NEURAL_NETWORK* nn);
    void* render_thread(void* arg);
}
