#pragma once

#include <string>
#include <vector>

namespace preprocessing {

class DataCleaner {
public:
    /**
     * 读取 CSV 文件，返回数据矩阵（行优先）并输出表头列名
     * @param filePath 文件路径
     * @param headers 输出：列名列表
     * @param data 输出：二维数据矩阵
     */
    static void readCSV(const std::string& filePath,
                        std::vector<std::string>& headers,
                        std::vector<std::vector<double>>& data);

    /**
     * 将数据矩阵写入 CSV 文件，首行为表头
     * @param filePath 输出文件路径
     * @param headers 列名列表
     * @param data 数据矩阵
     */
    static void writeCSV(const std::string& filePath,
                         const std::vector<std::string>& headers,
                         const std::vector<std::vector<double>>& data);

    /**
     * 基于 Z 分数去除指定列的异常值
     * @param data 输入数据矩阵
     * @param colIndex 要检测的列索引
     * @param zThreshold Z 分数阈值
     * @return 去除异常行后的新矩阵
     */
    static std::vector<std::vector<double>> removeOutliers(const std::vector<std::vector<double>>& data,
                                                            size_t colIndex,
                                                            double zThreshold = 3.0);

    /**
     * 等频分箱
     * @param values 输入数值向量
     * @param numBins 分箱数量
     * @return 每个值所属的分箱索引
     */
    static std::vector<int> equalFrequencyBinning(const std::vector<double>& values,
                                                   int numBins);

    /**
     * 在两个维度上先分箱再去除异常值
     * @param data 输入数据矩阵
     * @param colX 第一个分箱维度列索引
     * @param colY 第二个分箱维度列索引
     * @param numBins 分箱数
     * @param zThreshold Z 分数阈值
     * @return 去除异常后的新矩阵
     */
    static std::vector<std::vector<double>> removeOutliersByBinning(const std::vector<std::vector<double>>& data,
                                                                    size_t colX,
                                                                    size_t colY,
                                                                    int numBins,
                                                                    double zThreshold = 3.0);
};

} // namespace preprocessing