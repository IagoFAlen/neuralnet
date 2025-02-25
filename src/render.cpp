#include "globals.hpp"
#include "render.hpp"

#include <unistd.h> // For usleep
#include <cmath> // For fabs, cos, sin
#include <vector> // For storing loss values
#include <algorithm> // For std::max_element

namespace render {
    float cameraX = 0.0f;
    float cameraY = 0.0f;
    float zoom = 1.0f;

    // Store loss values for plotting
    std::vector<float> lossHistory;
    const size_t maxLossHistory = 500; // Maximum number of loss values to store

    void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width, 0, height, -1, 1);
        glMatrixMode(GL_MODELVIEW);
    }

    void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS || action == GLFW_REPEAT) {
            if (key == GLFW_KEY_S) cameraY += 10.0f;
            if (key == GLFW_KEY_W) cameraY -= 10.0f;
            if (key == GLFW_KEY_A) cameraX += 10.0f;
            if (key == GLFW_KEY_D) cameraX -= 10.0f;
            if (key == GLFW_KEY_E) zoom *= 1.1f;
            if (key == GLFW_KEY_Q) zoom /= 1.1f;
        }
    }

    void init_opengl() {
        if (!glfwInit()) exit(EXIT_FAILURE);
    
        // Request 4x MSAA before creating the window
        glfwWindowHint(GLFW_SAMPLES, 4);
    
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
    
        // Enable MSAA
        glEnable(GL_MULTISAMPLE);
    
        // Enable smooth lines
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetKeyCallback(glfwGetCurrentContext(), key_callback);
    }

    void render_loss_plot(int windowWidth, int windowHeight) {
        if (lossHistory.empty()) return;

        // Define the plot area (smaller and positioned at the bottom-right corner)
        float plotWidth = windowWidth * 0.25f; // 25% of the window width
        float plotHeight = windowHeight * 0.2f; // 20% of the window height
        float plotX = windowWidth - plotWidth - 20.0f; // Add some margin
        float plotY = 20.0f; // Add some margin

        // Draw the plot background
        glColor4f(0.1f, 0.1f, 0.1f, 0.8f); // Dark semi-transparent background
        glBegin(GL_QUADS);
        glVertex2f(plotX, plotY);
        glVertex2f(plotX + plotWidth, plotY);
        glVertex2f(plotX + plotWidth, plotY + plotHeight);
        glVertex2f(plotX, plotY + plotHeight);
        glEnd();

        // Find the maximum loss value for scaling
        float maxLoss = *std::max_element(lossHistory.begin(), lossHistory.end());
        if (maxLoss <= 0.0f) maxLoss = 1.0f; // Avoid division by zero

        // Draw the loss curve (white line)
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f); // White line
        glLineWidth(2.0f);
        glBegin(GL_LINE_STRIP);
        for (size_t i = 0; i < lossHistory.size(); i++) {
            float x = plotX + (float)i / (float)lossHistory.size() * plotWidth;
            float y = plotY + (lossHistory[i] / maxLoss) * plotHeight;
            glVertex2f(x, y);
        }
        glEnd();
    }

    void render_network(NEURAL_NETWORK* nn) {
        glClear(GL_COLOR_BUFFER_BIT);
    
        int windowWidth, windowHeight;
        glfwGetFramebufferSize(glfwGetCurrentContext(), &windowWidth, &windowHeight);
    
        // Calculate maximum number of neurons across all layers
        int maxNeurons = 0;
        for (LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next) {
            maxNeurons = std::max(maxNeurons, currentLayer->numNeurons);
        }
    
        // Calculate dynamic layer spacing
        float layerSpacing = windowWidth / (float)(nn->layersInfo->size + 1);
        float spacingFactor = 3.5f;
        layerSpacing *= spacingFactor;
        float neuronSpacingFactor = std::max(1.0f, (float)maxNeurons / spacingFactor / (float)(nn->layersInfo->size + 1));
        layerSpacing = layerSpacing * (1 + (neuronSpacingFactor / 4.0f));
    
        // Calculate neuron size based on layer spacing
        float neuronSize = 12.0f + (layerSpacing / (float)((nn->layersInfo->size + 1) * 10) );
        float baseNeuronSpacing = 100.0f;
    
        glPushMatrix();
        glTranslatef(cameraX, cameraY, 0.0f);
        glScalef(zoom, zoom, 1.0f);
    
        for (LAYER* currentLayer = nn->inputLayer; currentLayer != NULL; currentLayer = currentLayer->next) {
            // Calculate dynamic neuron vertical spacing based on neuron size
            float neuronSpacing = baseNeuronSpacing + (100.0f / neuronSize); // Adjust the constant as needed
    
            for (NEURON* currentNeuron = currentLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next) {
                float x = (currentLayer->id + 1) * layerSpacing;
                float y = (float)(windowHeight / 2) - ((float)currentLayer->numNeurons * neuronSpacing / 2.0f) + ((float)currentNeuron->id * neuronSpacing);
    
                float alpha = currentNeuron->activation;
                if (alpha < 0.1f)
                    alpha = 0.1f;
                glColor4f(1.0f, 1.0f, 1.0f, alpha);
    
                glBegin(GL_TRIANGLE_FAN);
                for (int angle = 0; angle < 360; angle += 10) {
                    float rad = angle * (3.14159f / 180.0f);
                    glVertex2f(x + cos(rad) * neuronSize, y + sin(rad) * neuronSize);
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
    
        glPopMatrix();
    
        render_loss_plot(windowWidth, windowHeight);
    }

    void* render_thread(void* arg) {
        NEURAL_NETWORK* nn = (NEURAL_NETWORK*)arg;
        init_opengl();

        while (!glfwWindowShouldClose(glfwGetCurrentContext())) {
            pthread_mutex_lock(&nn_mutex);

            // Simulate adding a new loss value (replace with actual loss calculation)
            static float simulatedLoss = 1.0f;
            simulatedLoss *= 0.99f; // Simulate decreasing loss
            lossHistory.push_back(simulatedLoss);
            if (lossHistory.size() > maxLossHistory) {
                lossHistory.erase(lossHistory.begin()); // Remove the oldest value
            }

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