#pragma once

#include <string>
#include <cstdint>

struct BaggingOptions {
    std::string dataPath;        
    int         numTrees;        
    double      sampleRatio;     
    int         maxDepth;        
    int         minSamplesLeaf;  
    std::string criterion;       
    std::string splitMethod;     
    std::string prunerType;      
    double      prunerParam;     
    uint32_t    seed;           
};


void runBaggingApp(const BaggingOptions& opts);
