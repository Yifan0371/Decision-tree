// =============================================================================
// include/functions/io/DataIO.hpp - 优化版本
// =============================================================================
#pragma once

#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <algorithm>

class DataIO {
public:
    // **核心方法**
    std::pair<std::vector<double>, std::vector<double>>
    readCSV(const std::string& filename, int& rowLength);

    void writeResults(const std::vector<double>& results,
                      const std::string& filename);

    // **新增：批量处理方法（用于大文件）**
    bool readCSVBatch(const std::string& filename, 
                      std::vector<double>& flattenedFeatures,
                      std::vector<double>& labels,
                      int& rowLength,
                      size_t batchSize = 10000,
                      size_t skipRows = 0);

    // **新增：并行写入方法**
    void writeResultsParallel(const std::vector<double>& results,
                              const std::string& filename,
                              size_t chunkSize = 10000);

    // **新增：内存映射读取方法（用于超大文件）**
    bool readCSVMemoryMapped(const std::string& filename,
                             std::vector<double>& flattenedFeatures,
                             std::vector<double>& labels,
                             int& rowLength);

    // **新增：数据验证方法**
    bool validateData(const std::vector<double>& flattenedFeatures,
                      const std::vector<double>& labels,
                      int rowLength);

    // **新增：获取文件统计信息**
    struct FileStats {
        size_t totalRows;
        size_t totalFeatures;
        size_t estimatedMemoryMB;
        bool hasHeader;
    };

    FileStats getFileStats(const std::string& filename) const;

    // **新增：流式读取接口（用于超大数据集）**
    class CSVReader {
    public:
        explicit CSVReader(const std::string& filename);
        ~CSVReader();
        
        bool hasNext() const;
        bool readNext(std::vector<double>& features, double& label);
        void reset();
        
    private:
        class Impl;
        std::unique_ptr<Impl> pImpl_;
    };

    std::unique_ptr<CSVReader> createReader(const std::string& filename);
};