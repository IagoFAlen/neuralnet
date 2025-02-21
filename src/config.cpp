#include <fstream>
#include <cstring>
#include <sstream>
#include <filesystem>
#include <cstring>

#include "config.hpp"
#include "neuralnetwork.hpp"
#include "utils.hpp"

using namespace std;
using namespace config;
using namespace neuralnets;
using namespace utils;

namespace fs = filesystem;

#define ANSI_COLOR_BRIGHT_YELLOW "\x1b[93m"
#define ANSI_COLOR_BRIGHT_RED "\x1b[91m"

namespace config
{
    NEURAL_NETWORK *initialize(unsigned int id, LIST_INFO *numNeuronsPerLayerList, int argc, char *argv[])
    {
        int num_layers = 0;
        bool found_l = false, found_n = false, found_lr = false, found_lambda = false, found_epochs = false;

        double learning_rate = 0.01; // Default learning rate
        double lambda = 0.001;       // Default lambda
        int epochs = 1000;           // Default epochs

        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "-l") == 0)
            {
                num_layers = atoi(argv[i + 1]);
                i++;
                found_l = true;
            }
            else if (strcmp(argv[i], "-n") == 0)
            {
                if (num_layers == 0)
                {
                    std::cerr << "Error: -n must be preceded by -l" << std::endl;
                    exit(1);
                }
                for (int j = 0; j < num_layers; j++)
                {
                    if (i + j + 1 >= argc || !isdigit(*argv[i + j + 1]))
                    {
                        std::cerr << "Error: Invalid number of neurons or missing value after -n" << std::endl;
                        exit(1);
                    }
                    ds_list::push(numNeuronsPerLayerList, atoi(argv[i + j + 1]));
                }
                i += num_layers;
                found_n = true;
            }
            else if (strcmp(argv[i], "--lr") == 0)
            {
                learning_rate = atof(argv[i + 1]);
                i++;
                found_lr = true;
            }
            else if (strcmp(argv[i], "--lambda") == 0)
            {
                lambda = atof(argv[i + 1]);
                i++;
                found_lambda = true;
            }
            else if (strcmp(argv[i], "--epochs") == 0)
            {
                epochs = atoi(argv[i + 1]);
                i++;
                found_epochs = true;
            }
        }

        if (!found_l || !found_n)
        {
            cerr << ANSI_COLOR_BRIGHT_RED << "Error: -l and -n are required arguments." << endl;
            exit(1);
        }
        if (!found_lr)
        {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --lr not specified, using default value: " << learning_rate << endl;
        }
        if (!found_lambda)
        {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --lambda not specified, using default value: " << lambda << endl;
        }
        if (!found_epochs)
        {
            cerr << ANSI_COLOR_BRIGHT_YELLOW << "Warning: --epochs not specified, using default value: " << epochs << endl;
        }

        NEURAL_NETWORK *newNeuralNetwork = create_neural_network(id, numNeuronsPerLayerList, learning_rate, lambda, epochs);
        return newNeuralNetwork;
    }

    void initialize_neurons(NEURAL_NETWORK *nn, LIST_INFO *inputList, LIST_INFO *targetList, string line)
    {
        string tempStringTrain = "";
        double trainData;

        stringstream inputString(line);
        for (int i = 0; i < (nn->layersInfo->list->value + nn->layersInfo->lastBlock->value); i++)
        {
            getline(inputString, tempStringTrain, ',');
            trainData = atof(tempStringTrain.c_str());
            if (i < nn->layersInfo->list->value)
                push(inputList, trainData);
            else
            {
                push(targetList, trainData);
            }

            string tempStringTrain = "";
        }

        int inputIndex = 0;

        for (NEURON *currentNeuron = nn->inputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next)
        {
            double normalizedInput = (get_value_by_index(inputList->list, inputIndex) / 100.00);
            currentNeuron->neuronValue = normalizedInput;
            currentNeuron->activation = currentNeuron->neuronValue;
            inputIndex++;
        }

        int targetIndex = 0;
        for (NEURON *currentNeuron = nn->outputLayer->neurons; currentNeuron != NULL; currentNeuron = currentNeuron->next)
        {
            currentNeuron->target = get_value_by_index(targetList->list, targetIndex);
            targetIndex++;
        }

        // utils::print_nn_io(nn);

        empty(inputList);
        empty(targetList);
    }

    void remove_file_train()
    {
        string current_directory = fs::current_path().string();

        fs::path log_path;

        fs::path train_log_path = fs::path(current_directory) / "visualize" / "loss_function.log";
        fs::path classify_log_path = fs::path(current_directory) / "visualize" / "classify.log";

        if (fs::exists(train_log_path))
            fs::remove(train_log_path);

        if (fs::exists(classify_log_path))
            fs::remove(classify_log_path);
    }

    void remove_file_predict()
    {
        string current_directory = fs::current_path().string();

        fs::path log_path;

        fs::path prediction_log_path = fs::path(current_directory) / "prediction" / "predict.log";

        if (fs::exists(prediction_log_path))
            fs::remove(prediction_log_path);
    }

    void save_loss_function(double lossFunction, char type)
    {
        string current_directory = fs::current_path().string();

        fs::path log_path;

        if (type == 'T')
        {
            log_path = fs::path(current_directory) / "visualize" / "loss_function.log";
        }
        else
        {
            log_path = fs::path(current_directory) / "visualize" / "classify.log";
        }

        ofstream file(log_path, ios::app);

        if (!file.is_open())
        {
            perror("Error opening log file");
            return;
        }

        file << lossFunction << "\n";

        file.close();
    }

    void save_predictions(double targetIndex, double predictionIndex)
    {
        string current_directory = fs::current_path().string();

        fs::path log_path;

        log_path = fs::path(current_directory) / "prediction" / "predict.log";

        ofstream file(log_path, ios::app);

        if (!file.is_open())
        {
            perror("Error opening log file");
            return;
        }

        file << targetIndex << "," << predictionIndex << "\n";

        file.close();
    }

    void train_with_epochs_randomly(NEURAL_NETWORK *nn, string filePath, bool saving_mode)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);

        if (!saving_mode)
            remove_file_train();

        if (!trainFile.is_open())
        {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";
        while (getline(trainFile, line))
        {
            file_list::push(lines, line);
            line = "";
        }

        trainFile.close();

        int index = 0;
        for (int epoch = 0; epoch < nn->epochs; ++epoch)
        {
            LIST_INFO *inputList = new LIST_INFO();
            LIST_INFO *targetList = new LIST_INFO();
            nn->learningRate *= 0.99; // Learning rate decay

            for (int i = 0; i < lines->size; i++)
            {
                int randomIndex = rand() % lines->size; // Randomly select a sample
                string currentLine = file_list::get_line_by_index(lines->file_list, randomIndex);

                initialize_neurons(nn, inputList, targetList, currentLine);
                neuralnets::feed_forward(nn);
                neuralnets::backpropagation(nn);
                save_loss_function(nn->lossFunction, 'T');

                print_train(index, nn->epochs * lines->size);
                index++;
            }
        }
    }

    void train_with_epochs(NEURAL_NETWORK *nn, string filePath, bool saving_mode)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream trainFile(filePath);

        if (!saving_mode)
            remove_file_train();

        if (!trainFile.is_open())
        {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";

        while (getline(trainFile, line))
        {
            file_list::push(lines, line);
            line = "";
        }

        trainFile.close();

        int index = 0;
        for (int epoch = 0; epoch < nn->epochs; ++epoch)
        {
            LIST_INFO *inputList = new LIST_INFO();
            LIST_INFO *targetList = new LIST_INFO();
            nn->learningRate *= 0.99;
            for (int i = 0; i < lines->size; i++)
            {
                string currentLine = file_list::get_line_by_index(lines->file_list, i);
                initialize_neurons(nn, inputList, targetList, currentLine);
                neuralnets::feed_forward(nn);
                neuralnets::backpropagation(nn);
                save_loss_function(nn->lossFunction, 'T');
                // cout << currentLine << endl;
                print_train(index, nn->epochs * lines->size);
                index++;
            }
        }
    }

    void classify(NEURAL_NETWORK *nn, string filePath)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream classifyFile(filePath);

        if (!classifyFile.is_open())
        {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";

        while (getline(classifyFile, line))
        {
            file_list::push(lines, line);
            line = "";
        }

        classifyFile.close();

        LIST_INFO *inputList = new LIST_INFO();
        LIST_INFO *targetList = new LIST_INFO();

        for (int i = 0; i < lines->size; i++)
        {
            string currentLine = file_list::get_line_by_index(lines->file_list, i);
            initialize_neurons(nn, inputList, targetList, currentLine);
            neuralnets::feed_forward(nn);
            neuralnets::loss_function(nn);
            save_loss_function(nn->lossFunction, 'C');
            // cout << currentLine << endl;
        }
    }

    void predict(NEURAL_NETWORK *nn, string filePath)
    {
        FILE_LIST_INFO *lines = new FILE_LIST_INFO();

        ifstream classifyFile(filePath);

        remove_file_predict();
        if (!classifyFile.is_open())
        {
            cout << "Error opening file: " << filePath << endl;
            return;
        }

        string line = "";

        while (getline(classifyFile, line))
        {
            file_list::push(lines, line);
            line = "";
        }

        classifyFile.close();

        LIST_INFO *inputList = new LIST_INFO();
        LIST_INFO *targetList = new LIST_INFO();

        for (int i = 0; i < lines->size; i++)
        {
            string currentLine = file_list::get_line_by_index(lines->file_list, i);
            initialize_neurons(nn, inputList, targetList, currentLine);
            neuralnets::feed_forward(nn);
            neuralnets::predict(nn);
            // cout << currentLine << endl;
        }
    }

    void save_neural_network(NEURAL_NETWORK *nn, const string &filePath)
    {
        ofstream file(filePath);
        if (!file.is_open())
        {
            cerr << "Error: Unable to open file for saving neural network." << endl;
            return;
        }

        // Save basic network information
        file << nn->id << " " << nn->learningRate << " " << nn->lambda << " " << nn->epochs << endl;

        // Save layer information
        for (LAYER *layer = nn->inputLayer; layer != nullptr; layer = layer->next)
        {
            file << layer->id << " " << layer->numNeurons << endl;

            // Save neuron information
            for (NEURON *neuron = layer->neurons; neuron != nullptr; neuron = neuron->next)
            {
                file << neuron->id << " " << neuron->bias << endl;

                // Save connection information
                for (CONNECTION *conn = neuron->connections; conn != nullptr; conn = conn->next)
                {
                    file << conn->id << " " << conn->weight << " " << conn->afterwardNeuron->id << endl;
                }
            }
        }

        file.close();
    }

    NEURAL_NETWORK *load_neural_network(const string &filePath)
    {
        ifstream file(filePath);
        if (!file.is_open())
        {
            cerr << "Error: Unable to open file for loading neural network." << endl;
            return nullptr;
        }

        NEURAL_NETWORK *nn = new NEURAL_NETWORK();
        unsigned int id;
        double learningRate, lambda;
        int epochs;

        // Load basic network information
        file >> id >> learningRate >> lambda >> epochs;
        nn->id = id;
        nn->learningRate = learningRate;
        nn->lambda = lambda;
        nn->epochs = epochs;

        // Load layer information
        LAYER *prevLayer = nullptr;
        while (file.peek() != EOF)
        {
            unsigned int layerId;
            int numNeurons;
            file >> layerId >> numNeurons;

            LAYER *layer = create_layer(numNeurons, layerId);
            if (prevLayer != nullptr)
            {
                prevLayer->next = layer;
                layer->previous = prevLayer;
            }
            else
            {
                nn->inputLayer = layer;
            }
            prevLayer = layer;

            // Load neuron information
            for (int i = 0; i < numNeurons; ++i)
            {
                unsigned int neuronId;
                double bias;
                file >> neuronId >> bias;

                NEURON *neuron = create_neuron(neuronId, bias);
                if (layer->neurons == nullptr)
                {
                    layer->neurons = neuron;
                    layer->lastNeuron = neuron;
                }
                else
                {
                    layer->lastNeuron->next = neuron;
                    layer->lastNeuron = neuron;
                }

                // Load connection information
                while (file.peek() != '\n' && file.peek() != EOF)
                {
                    unsigned int connId;
                    double weight;
                    unsigned int destNeuronId;
                    file >> connId >> weight >> destNeuronId;

                    NEURON *destNeuron = layer->next->neurons;
                    while (destNeuron != nullptr && destNeuron->id != destNeuronId)
                    {
                        destNeuron = destNeuron->next;
                    }

                    if (destNeuron != nullptr)
                    {
                        create_connection(connId, neuron, weight, destNeuron);
                    }
                }
            }
        }

        nn->outputLayer = prevLayer;
        file.close();
        return nn;
    }
}