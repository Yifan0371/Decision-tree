
// =============================================================================
// src/regression_boosting_main.cpp
// =============================================================================
#include "boosting/app/RegressionBoostingApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        RegressionBoostingOptions opts = parseRegressionCommandLine(argc, argv);
        
        if (opts.verbose) {
            std::cout << "=== GBRT Configuration ===" << std::endl;
            std::cout << "Data: " << opts.dataPath << std::endl;
            std::cout << "Loss: " << opts.lossFunction << std::endl;
            std::cout << "Iterations: " << opts.numIterations << std::endl;
            std::cout << "Learning Rate: " << opts.learningRate << std::endl;
            std::cout << std::endl;
        }
        
        runRegressionBoostingApp(opts);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}