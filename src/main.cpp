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


    NEURAL_NETWORK* nn = new NEURAL_NETWORK();

    nn = config::initialize(1, numNeuronsPerLayerList, argc, argv, 0.5);

    return 0;
}