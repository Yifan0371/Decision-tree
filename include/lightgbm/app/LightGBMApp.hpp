#pragma once

#include "lightgbm/core/LightGBMConfig.hpp"
#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include <string>
#include <memory>


struct LightGBMAppOptions {
    std::string dataPath = "../data/data_clean/cleaned_data.csv";
    std::string objective = "regression";
    
    
    int numIterations = 100;
    double learningRate = 0.1;
    int maxDepth = -1;
    int numLeaves = 31;
    int minDataInLeaf = 20;
    
    
    double topRate = 0.2;
    double otherRate = 0.1;
    
    
    int maxBin = 255;
    double maxConflictRate = 0.0;
    bool enableFeatureBundling = true;
    bool enableGOSS = true;
    
    
    bool verbose = true;
    int earlyStoppingRounds = 0;
    double tolerance = 1e-7;
    double valSplit = 0.2;
    
    
    double lambda = 0.0;
    double minSplitGain = 0.0;
    
    std::string splitMethod = "histogram_ew";
    int histogramBins = 255;
    std::string adaptiveRule = "sturges";
    int minSamplesPerBin = 5;
    int maxAdaptiveBins = 128;
    double variabilityThreshold = 0.1;
    bool enableSIMD = true;
};


void runLightGBMApp(const LightGBMAppOptions& options);


std::unique_ptr<LightGBMTrainer> createLightGBMTrainer(const LightGBMAppOptions& options);


LightGBMAppOptions parseLightGBMCommandLine(int argc, char** argv);


void printLightGBMModelSummary(const LightGBMTrainer* trainer, const LightGBMAppOptions& options);
