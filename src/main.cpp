#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>
#include <filesystem>

#include "neuralnetwork.hpp"
#include "config.hpp"
#include "list.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;
using namespace config;

namespace fs = std::filesystem;

int main(int argc, char *argv[]){
    srand(time(NULL));
    LIST_INFO* numNeuronsPerLayerList = new LIST_INFO();
    
    string current_directory = fs::current_path().string();
    string TRAIN_FILE_PATH = current_directory + "/training/training.csv";
    string CLASSIFY_FILE_PATH = current_directory + "/classification/testing.csv";

    NEURAL_NETWORK* nn = new NEURAL_NETWORK();

    nn = config::initialize(1, numNeuronsPerLayerList, argc, argv);

    bool saving_mode = false;
    //config::train_with_epochs_randomly(nn, TRAIN_FILE_PATH, EPOCHS, saving_mode, double lambda);
    config::train_with_epochs(nn, TRAIN_FILE_PATH, saving_mode);
    config::classify(nn, CLASSIFY_FILE_PATH);

    return 0;
}