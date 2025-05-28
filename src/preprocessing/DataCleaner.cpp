#include "preprocessing/DataCleaner.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace preprocessing {

void DataCleaner::readCSV(const std::string& filePath,
                          std::vector<std::string>& headers,
                          std::vector<std::vector<double>>& data) {
    std::ifstream in(filePath);
    if (!in.is_open()) {
        throw std::runtime_error("无法打开文件: " + filePath);
    }
    std::string line;
    // 读取表头
    if (!std::getline(in, line)) return;
    headers.clear();
    std::stringstream ssHead(line);
    std::string col;
    while (std::getline(ssHead, col, ',')) {
        headers.push_back(col);
    }
    // 读取数据行
    data.clear();
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        if (!row.empty()) data.push_back(std::move(row));
    }
    in.close();
}

void DataCleaner::writeCSV(const std::string& filePath,
                           const std::vector<std::string>& headers,
                           const std::vector<std::vector<double>>& data) {
    std::ofstream out(filePath);
    if (!out.is_open()) {
        throw std::runtime_error("无法写入文件: " + filePath);
    }
    // 写入表头
    for (size_t i = 0; i < headers.size(); ++i) {
        out << headers[i] << (i + 1 < headers.size() ? ',' : '\n');
    }
    // 写入数据行
    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j] << (j + 1 < row.size() ? ',' : '\n');
        }
    }
    out.close();
}

std::vector<std::vector<double>> DataCleaner::removeOutliers(
    const std::vector<std::vector<double>>& data,
    size_t colIndex,
    double zThreshold) {
    // 计算均值与标准差
    std::vector<double> colVals;
    colVals.reserve(data.size());
    for (const auto& row : data) {
        if (colIndex < row.size())
            colVals.push_back(row[colIndex]);
    }
    double mean = std::accumulate(colVals.begin(), colVals.end(), 0.0) / colVals.size();
    double var = 0.0;
    for (double v : colVals) var += (v - mean) * (v - mean);
    var /= colVals.size();
    double stddev = std::sqrt(var);

    std::vector<std::vector<double>> cleaned;
    for (const auto& row : data) {
        double v = row[colIndex];
        double z = std::abs((v - mean) / (stddev + 1e-12));
        if (z <= zThreshold) {
            cleaned.push_back(row);
        }
    }
    return cleaned;
}

std::vector<int> DataCleaner::equalFrequencyBinning(
    const std::vector<double>& values,
    int numBins) {
    int n = static_cast<int>(values.size());
    std::vector<std::pair<double,int>> sorted;
    sorted.reserve(n);
    for (int i = 0; i < n; ++i) sorted.emplace_back(values[i], i);
    std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b){ return a.first < b.first; });

    std::vector<int> bins(n);
    int baseSize = n / numBins;
    int rem = n % numBins;
    int idx = 0;
    for (int b = 0; b < numBins; ++b) {
        int thisSize = baseSize + (b < rem ? 1 : 0);
        for (int k = 0; k < thisSize; ++k) {
            bins[sorted[idx].second] = b;
            ++idx;
        }
    }
    return bins;
}

std::vector<std::vector<double>> DataCleaner::removeOutliersByBinning(
    const std::vector<std::vector<double>>& data,
    size_t colX,
    size_t colY,
    int numBins,
    double zThreshold) {
    // 提取两个维度
    std::vector<double> valsX, valsY;
    valsX.reserve(data.size());
    valsY.reserve(data.size());
    for (auto &row : data) {
        valsX.push_back(row[colX]);
        valsY.push_back(row[colY]);
    }
    auto binsX = equalFrequencyBinning(valsX, numBins);
    auto binsY = equalFrequencyBinning(valsY, numBins);

    std::vector<std::vector<double>> result;
    // 对每个 bin 分别计算并去除异常
    for (int b = 0; b < numBins; ++b) {
        // 收集 bin 内索引
        std::vector<int> idxs;
        std::vector<double> perf;
        for (size_t i = 0; i < data.size(); ++i) {
            if (binsX[i] == b || binsY[i] == b) {
                idxs.push_back(i);
                perf.push_back(data[i].back());
            }
        }
        if (idxs.empty()) continue;
        double mean = std::accumulate(perf.begin(), perf.end(), 0.0) / perf.size();
        double var = 0.0;
        for (double v : perf) var += (v - mean) * (v - mean);
        var /= perf.size();
        double stddev = std::sqrt(var);
        // 保留正常
        for (auto i : idxs) {
            double z = std::abs((perf[i - idxs.front()] - mean) / (stddev + 1e-12));
            if (z <= zThreshold) {
                result.push_back(data[i]);
            }
        }
    }
    return result;
}

} // namespace preprocessing
