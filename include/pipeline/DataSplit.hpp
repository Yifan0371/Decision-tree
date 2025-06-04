#pragma once

#include <vector>

struct DataParams {
    std::vector<double> X_train;
    std::vector<double> y_train;
    std::vector<double> X_test;
    std::vector<double> y_test;
    int rowLength; 
};


bool splitDataset(const std::vector<double>& X,
                  const std::vector<double>& y,
                  int rowLength,
                  DataParams& out);
