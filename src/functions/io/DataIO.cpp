#include "functions/io/DataIO.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

std::pair<std::vector<double>, std::vector<double>>
DataIO::readCSV(const std::string& filename, int& rowLength) {
    std::vector<double> flattenedFeatures;
    std::vector<double> labels;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return {flattenedFeatures, labels};
    }

    std::string line;
    bool headerSkipped = false;
    rowLength = 0;

    while (std::getline(file, line)) {
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        if (!row.empty()) {
            labels.push_back(row.back());
            row.pop_back();
            flattenedFeatures.insert(flattenedFeatures.end(),
                                     row.begin(), row.end());
            if (rowLength == 0) {
                rowLength = static_cast<int>(row.size()) + 1;
            }
        }
    }

    file.close();
    return {flattenedFeatures, labels};
}

void DataIO::writeResults(const std::vector<double>& results,
                          const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    for (const auto& r : results) {
        file << r << "\n";
    }
    file.close();
}
