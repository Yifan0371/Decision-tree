#pragma once

#include "../trainer/GBRTTrainer.hpp"
#include <string>
#include <memory>


struct RegressionBoostingOptions {
    
    std::string dataPath;              
    std::string lossFunction = "squared"; 
    
    
    int numIterations = 100;           
    double learningRate = 0.1;         
    int maxDepth = 6;                  
    int minSamplesLeaf = 1;            
    std::string criterion = "mse";     
    std::string splitMethod = "exhaustive";  
    std::string prunerType = "none";   
    double prunerParam = 0.0;          
    
    
    bool verbose = true;               
    int earlyStoppingRounds = 0;       
    double tolerance = 1e-7;           
    double valSplit = 0.2;             
    
    
    double huberDelta = 1.0;           
    double quantile = 0.5;             
    
    
    bool useLineSearch = false;        
    double subsample = 1.0;    
     
    bool enableDart = false;
    double dartDropRate = 0.1;
    bool dartNormalize = true;
    bool dartSkipDropForPrediction = false;
    std::string dartStrategy = "uniform";
    uint32_t dartSeed = 42;        
};


void runRegressionBoostingApp(const RegressionBoostingOptions& options);


std::unique_ptr<GBRTTrainer> createRegressionBoostingTrainer(const RegressionBoostingOptions& options);


RegressionBoostingOptions parseRegressionCommandLine(int argc, char** argv);


void printRegressionModelSummary(const GBRTTrainer* trainer, const RegressionBoostingOptions& options);
