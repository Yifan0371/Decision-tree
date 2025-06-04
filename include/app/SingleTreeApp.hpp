#pragma once

#include <string>


struct ProgramOptions {
    std::string dataPath;        
    int         maxDepth;        
    int         minSamplesLeaf;  
    std::string criterion;       
    std::string splitMethod;     
    
    
    std::string prunerType;      
    double      prunerParam;     
    double      valSplit;        
};


void runSingleTreeApp(const ProgramOptions&);
