#include "globals.hpp"
#include "render.hpp"

#include <unistd.h> // For usleep
#include <cmath> // For fabs, cos, sin

namespace render {
    void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width, 0, height, -1, 1);
        glMatrixMode(GL_MODELVIEW);
    }

    void init_opengl() {
        if (!glfwInit()) exit(EXIT_FAILURE);

        GLFWwindow* window = glfwCreateWindow(800, 600, "Neural Network Visualizer", NULL, NULL);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width, 0, height, -1, 1);
        glMatrixMode(GL_MODELVIEW);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    }

    void render_network(NEURAL_NETWORK* nn) {
        glClear(GL_COLOR_BUFFER_BIT);

        int windowWidth, windowHeight;
        glfwGetFramebufferSize(glfwGetCurrentContext(), &windowWidth, &windowHeight);

        float layerSpacing = (float)windowWidth / ((float)nn->layersInfo->size + 1.0f);
        float neuronSpacing = 40.0f;

        for (LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next) {
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                float x = (currentLayer->id + 1) * layerSpacing;
                float y = (float)(windowHeight / 2) - ((float)currentLayer->numNeurons * neuronSpacing / 2.0f) + ((float)currentNeuron->id * neuronSpacing);

                float alpha = currentNeuron->activation;
                glColor4f(1.0f, 1.0f, 1.0f, alpha);

                glBegin(GL_TRIANGLE_FAN);
                for (int angle = 0; angle < 360; angle += 10) {
                    float rad = angle * (3.14159f / 180.0f);
                    glVertex2f(x + cos(rad) * 12.0f, y + sin(rad) * 12.0f);
                }
                glEnd();

                for (CONNECTION* currentConnection = currentNeuron->connections; currentConnection != NULL; currentConnection = currentConnection->next) {
                    float dst_x = (currentLayer->next->id + 1) * layerSpacing;
                    float dst_y = (float)(windowHeight / 2) - ((float)currentLayer->next->numNeurons * neuronSpacing / 2.0f) + ((float)currentConnection->afterwardNeuron->id * neuronSpacing);

                    float weightAlpha = fabs(currentConnection->weight);
                    glColor4f(1.0f, 1.0f, 1.0f, weightAlpha);

                    glLineWidth(2.0f);
                    glBegin(GL_LINES);
                    glVertex2f(x, y);
                    glVertex2f(dst_x, dst_y);
                    glEnd();
                }
            }
        }
    }

    void* render_thread(void* arg) {
        NEURAL_NETWORK* nn = (NEURAL_NETWORK*)arg;
        init_opengl();

        while (!glfwWindowShouldClose(glfwGetCurrentContext())) {
            pthread_mutex_lock(&nn_mutex);
            render_network(nn);
            glfwSwapBuffers(glfwGetCurrentContext());
            pthread_mutex_unlock(&nn_mutex);
            glfwPollEvents();
            usleep(16000);
        }

        glfwTerminate();
        return NULL;
    }
}