#pragma once

#include <vector>
#include <string>
#include <utility>

class DataIO {
public:
    std::pair<std::vector<double>, std::vector<double>>
    readCSV(const std::string& filename, int& rowLength);

    void writeResults(const std::vector<double>& results,
                      const std::string& filename);
};
