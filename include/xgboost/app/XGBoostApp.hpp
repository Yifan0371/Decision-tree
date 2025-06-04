#pragma once

#include "xgboost/core/XGBoostConfig.hpp"
#include "xgboost/trainer/XGBoostTrainer.hpp"
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"


struct XGBoostAppOptions {
    std::string dataPath = "../data/data_clean/cleaned_data.csv";
    std::string objective = "reg:squarederror";
    
    
    int numRounds = 100;
    double eta = 0.3;
    int maxDepth = 6;
    int minChildWeight = 1;
    double lambda = 1.0;
    double gamma = 0.0;
    double subsample = 1.0;
    double colsampleByTree = 1.0;
    
    
    bool verbose = true;
    int earlyStoppingRounds = 0;
    double tolerance = 1e-7;
    double valSplit = 0.2;
    
    
    bool useApproxSplit = false;
    int maxBins = 256;
};


void runXGBoostApp(const XGBoostAppOptions& options);


std::unique_ptr<XGBoostTrainer> createXGBoostTrainer(const XGBoostAppOptions& options);


XGBoostAppOptions parseXGBoostCommandLine(int argc, char** argv);


void printXGBoostModelSummary(const XGBoostTrainer* trainer, const XGBoostAppOptions& options);
