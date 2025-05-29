#ifndef PIPELINE_DATASPLIT_HPP
#define PIPELINE_DATASPLIT_HPP

#include <vector>

struct DataParams {
    std::vector<double> X_train;
    std::vector<double> y_train;
    std::vector<double> X_test;
    std::vector<double> y_test;
    int rowLength; // number of features
};

/**
 * 按 80/20 划分 X（扁平化）和 y
 * @param X 输入特征扁平化数组
 * @param y 输入标签数组
 * @param rowLength 包括标签的列数（从 DataIO 获得）
 * @param out 输出的训练/测试集
 */
bool splitDataset(const std::vector<double>& X,
                  const std::vector<double>& y,
                  int rowLength,
                  DataParams& out);

#endif // PIPELINE_DATASPLIT_HPP
