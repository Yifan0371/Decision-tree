#include "preprocessing/DataCleaner.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    const std::string inDir  = "data/data_base";
    const std::string outDir = "data/data_clean";

    // 确保输出目录存在
    fs::create_directories(outDir);

    std::vector<std::string> headers;
    std::vector<std::vector<double>> data, cleaned;

    for (auto& entry : fs::directory_iterator(inDir)) {
        if (entry.path().extension() == ".csv") {
            std::string filename = entry.path().filename().string();
            std::string inPath  = entry.path().string();
            std::string outPath = outDir + "/cleaned_" + filename;

            try {
                // 读 CSV
                preprocessing::DataCleaner::readCSV(inPath, headers, data);

                // 全局剔除最后一列的异常值（Z 阈值 3.0）
                cleaned = preprocessing::DataCleaner::removeOutliers(data, headers.size()-1, 3.0);

                // 写入 clean 目录
                preprocessing::DataCleaner::writeCSV(outPath, headers, cleaned);

                std::cout << "Cleaned " << filename << " -> " << outPath << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << filename << ": " << e.what() << std::endl;
            }
        }
    }
    return 0;
}
