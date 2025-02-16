#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>

#include "neuralnetwork.hpp"
#include "config.hpp"
#include "list.hpp"

using namespace std;
using namespace neuralnets;
using namespace ds_list;
using namespace config;


int main(int argc, char *argv[]){
    srand(time(NULL));
    LIST_INFO* numNeuronsPerLayerList = new LIST_INFO();
    double LEARNING_RATE = 0.00001;
    string userEpochs = "";
    cout << "Training times: ";
    cin >> userEpochs;

    int EPOCHS = stoi(userEpochs);
    string TRAIN_FILE_PATH = "/home/john/playground/neuralnets/training/training.csv";

    NEURAL_NETWORK* nn = new NEURAL_NETWORK();

    nn = config::initialize(1, numNeuronsPerLayerList, argc, argv, LEARNING_RATE);

    //config::train_with_epochs_randomly(nn, TRAIN_FILE_PATH, EPOCHS);
    config::train_with_epochs(nn, TRAIN_FILE_PATH, EPOCHS);
    return 0;
}