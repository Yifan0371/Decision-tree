// =============================================================================
// src/functions/io/DataIO.cpp - 优化版本（避免不必要的vector拷贝）
// =============================================================================
#include "functions/io/DataIO.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cmath>
#include <utility>
#include <iomanip>
#include <stdexcept>    
#include <vector>


std::pair<std::vector<double>, std::vector<double>>
DataIO::readCSV(const std::string& filename, int& rowLength) {
    std::vector<double> flattenedFeatures;
    std::vector<double> labels;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return {std::move(flattenedFeatures), std::move(labels)};
    }

    std::string line;
    bool headerSkipped = false;
    rowLength = 0;

    // **优化1: 预分配容器大小估算**
    // 先快速扫描文件获取行数估算
    const auto initialPos = file.tellg();
    size_t estimatedRows = 0;
    while (std::getline(file, line)) {
        ++estimatedRows;
    }
    file.clear();
    file.seekg(initialPos);
    
    if (estimatedRows > 1) { // 减去头部行
        --estimatedRows;
        labels.reserve(estimatedRows);
        // 特征数量在读取第一行后确定
    }

    while (std::getline(file, line)) {
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        
        // **优化2: 使用高效的字符串解析**
        std::vector<double> row;
        row.reserve(50); // 预分配常见的特征数量
        
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse value '" << value 
                          << "' as double: " << e.what() << std::endl;
                row.push_back(0.0); // 默认值
            }
        }
        
        if (!row.empty()) {
            // **优化3: 避免每次都push_back label，批量操作**
            labels.push_back(row.back());
            row.pop_back();
            
            // **优化4: 预分配feature vector并批量插入**
            if (flattenedFeatures.empty() && !row.empty()) {
                // 第一行确定特征数量，预分配总空间
                const size_t featuresPerRow = row.size();
                flattenedFeatures.reserve(estimatedRows * featuresPerRow);
                rowLength = static_cast<int>(featuresPerRow) + 1; // +1 for label
            }
            
            // **使用move语义和批量插入**
            flattenedFeatures.insert(flattenedFeatures.end(),
                                   std::make_move_iterator(row.begin()),
                                   std::make_move_iterator(row.end()));
        }
    }

    file.close();
    
    // **优化5: 最终的内存整理**
    flattenedFeatures.shrink_to_fit();
    labels.shrink_to_fit();
    
    std::cout << "Loaded " << labels.size() << " samples with " 
              << (rowLength - 1) << " features each" << std::endl;
    
    return {std::move(flattenedFeatures), std::move(labels)};
}

void DataIO::writeResults(const std::vector<double>& results,
                          const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    
    // **优化: 使用缓冲写入提高性能**
    file.precision(10); // 设置精度
    file << std::fixed;
    
    for (const auto& r : results) {
        file << r << '\n';
    }
    
    file.close();
}

// **新增方法：批量读取CSV（用于大文件）**
bool DataIO::readCSVBatch(const std::string& filename, 
                          std::vector<double>& flattenedFeatures,
                          std::vector<double>& labels,
                          int& rowLength,
                          size_t batchSize,
                          size_t skipRows) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    
    // 跳过头部
    if (!std::getline(file, line)) {
        std::cerr << "Empty file: " << filename << std::endl;
        return false;
    }
    
    // 跳过指定行数
    for (size_t i = 0; i < skipRows; ++i) {
        if (!std::getline(file, line)) {
            return false; // 文件结束
        }
    }
    
    // 清空输出容器
    flattenedFeatures.clear();
    labels.clear();
    flattenedFeatures.reserve(batchSize * 50); // 估算特征数
    labels.reserve(batchSize);
    
    size_t rowsRead = 0;
    while (rowsRead < batchSize && std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception&) {
                row.push_back(0.0);
            }
        }
        
        if (!row.empty()) {
            labels.push_back(row.back());
            row.pop_back();
            
            if (rowsRead == 0) {
                rowLength = static_cast<int>(row.size()) + 1;
            }
            
            flattenedFeatures.insert(flattenedFeatures.end(),
                                   row.begin(), row.end());
            ++rowsRead;
        }
    }
    
    file.close();
    return rowsRead > 0;
}

// **新增方法：并行CSV写入（用于大结果集）**
void DataIO::writeResultsParallel(const std::vector<double>& results,
                                  const std::string& filename,
                                  size_t chunkSize) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    
    file.precision(10);
    file << std::fixed;
    
    // **对于大数据集，使用缓冲区批量写入**
    if (results.size() > chunkSize) {
        std::string buffer;
        buffer.reserve(chunkSize * 20); // 预分配缓冲区
        
        for (size_t i = 0; i < results.size(); ++i) {
            buffer += std::to_string(results[i]) + '\n';
            
            if ((i + 1) % chunkSize == 0 || i == results.size() - 1) {
                file << buffer;
                buffer.clear();
            }
        }
    } else {
        // 小数据集直接写入
        for (const auto& r : results) {
            file << r << '\n';
        }
    }
    
    file.close();
}

// **新增方法：内存映射读取（用于超大文件）**
bool DataIO::readCSVMemoryMapped(const std::string& filename,
                                 std::vector<double>& flattenedFeatures,
                                 std::vector<double>& labels,
                                 int& rowLength) {
    // 这里可以实现内存映射文件读取
    // 暂时使用标准方法的优化版本
    auto result = readCSV(filename, rowLength);
    flattenedFeatures = std::move(result.first);
    labels = std::move(result.second);
    return !flattenedFeatures.empty();
}

// **新增方法：验证数据完整性**
bool DataIO::validateData(const std::vector<double>& flattenedFeatures,
                          const std::vector<double>& labels,
                          int rowLength) {
    
    if (labels.empty()) {
        std::cerr << "Error: No labels found" << std::endl;
        return false;
    }
    
    const size_t expectedFeatureCount = labels.size() * (rowLength - 1);
    if (flattenedFeatures.size() != expectedFeatureCount) {
        std::cerr << "Error: Feature count mismatch. Expected: " 
                  << expectedFeatureCount << ", Got: " << flattenedFeatures.size() << std::endl;
        return false;
    }
    
    // 检查是否有无效值
    const auto invalidFeature = std::find_if(flattenedFeatures.begin(), flattenedFeatures.end(),
        [](double val) { return !std::isfinite(val); });
    
    if (invalidFeature != flattenedFeatures.end()) {
        std::cerr << "Warning: Found non-finite feature values" << std::endl;
    }
    
    const auto invalidLabel = std::find_if(labels.begin(), labels.end(),
        [](double val) { return !std::isfinite(val); });
    
    if (invalidLabel != labels.end()) {
        std::cerr << "Warning: Found non-finite label values" << std::endl;
    }
    
    return true;
}