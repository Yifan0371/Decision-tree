#ifndef APP_SINGLE_TREE_APP_HPP
#define APP_SINGLE_TREE_APP_HPP

#include <string>

struct ProgramOptions {
    std::string dataPath;
    int maxDepth;
    int minSamplesLeaf;
};

void runSingleTreeApp(const ProgramOptions& opts);

#endif // APP_SINGLE_TREE_APP_HPP
