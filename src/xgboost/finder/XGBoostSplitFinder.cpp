
// 修复 src/xgboost/finder/XGBoostSplitFinder.cpp
#include "xgboost/finder/XGBoostSplitFinder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>  // 添加调试输出
#include <iomanip>   // 用于设置输出精度


std::tuple<int, double, double> XGBoostSplitFinder::findBestSplit(
    const std::vector<double>& /* data */,
    int /* rowLength */,
    const std::vector<double>& /* labels */,
    const std::vector<int>& /* indices */,
    double /* currentMetric */,
    const ISplitCriterion& /* criterion */) const {
    
    // 对于XGBoost，这个接口不应该被调用
    std::cout << "WARNING: 旧接口被调用，应该使用findBestSplitXGB" << std::endl;
    return {-1, 0.0, 0.0};
}


std::tuple<int, double, double> XGBoostSplitFinder::findBestSplitXGB(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& indices,
    const XGBoostCriterion& xgbCriterion) const {
    
    if (indices.size() < 2) return {-1, 0.0, 0.0};
    
    // 计算父节点统计量
    auto [G_parent, H_parent] = computeGradHessStats(gradients, hessians, indices);
    
    // 添加调试输出
    static int callCount = 0;
    callCount++;
    if (callCount <= 3) {
        std::cout << "DEBUG: 父节点 G=" << G_parent << ", H=" << H_parent << std::endl;
        std::cout << "DEBUG: 开始搜索 " << rowLength << " 个特征" << std::endl;
    }
    
    // 检查最小子节点权重约束
    if (H_parent < minChildWeight_) {
        if (callCount <= 3) {
            std::cout << "DEBUG: 父节点权重不足: " << H_parent 
                      << " < " << minChildWeight_ << std::endl;
        }
        return {-1, 0.0, 0.0};
    }
    
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    
    // 预分配临时缓冲区
    std::vector<int> sortedIndices(indices.size());
    const double EPS = 1e-12;
    
    // 遍历所有特征
    for (int f = 0; f < rowLength; ++f) {
        // 复制并按特征值排序
        std::copy(indices.begin(), indices.end(), sortedIndices.begin());
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&](int a, int b) {
                      return data[a * rowLength + f] < data[b * rowLength + f];
                  });
        
        // 计算前缀统计量
        double G_left = 0.0, H_left = 0.0;
        int validSplits = 0;
        
        // 遍历所有可能的分裂点
        for (size_t i = 0; i < sortedIndices.size() - 1; ++i) {
            int idx = sortedIndices[i];
            G_left += gradients[idx];
            H_left += hessians[idx];
            
            double currentVal = data[sortedIndices[i] * rowLength + f];
            double nextVal = data[sortedIndices[i + 1] * rowLength + f];
            
            // 跳过相同特征值
            if (std::abs(nextVal - currentVal) < EPS) continue;
            
            // 计算右子节点统计量
            double G_right = G_parent - G_left;
            double H_right = H_parent - H_left;
            
            // 检查最小子节点权重约束
            if (H_left < minChildWeight_ || H_right < minChildWeight_) continue;
            
            // 计算分裂增益
            double gain = xgbCriterion.computeSplitGain(
                G_left, H_left, G_right, H_right, G_parent, H_parent, gamma_);
            
            validSplits++;
            
            // 调试输出（只输出前几个分裂）
            if (callCount <= 2 && f <= 1 && validSplits <= 5) {
                std::cout << "DEBUG: 特征" << f << " 分裂点" << validSplits 
                          << " G_L=" << std::fixed << std::setprecision(6) << G_left 
                          << " H_L=" << H_left
                          << " G_R=" << G_right << " H_R=" << H_right
                          << " gain=" << gain << " threshold=" << (0.5 * (currentVal + nextVal)) << std::endl;
            }
            
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = f;
                bestThreshold = 0.5 * (currentVal + nextVal);
                
                if (callCount <= 2) {
                    std::cout << "DEBUG: 找到更好分裂! 特征" << f 
                              << " 增益=" << std::fixed << std::setprecision(6) << gain << std::endl;
                }
            }
        }
        
        if (callCount <= 2) {
            std::cout << "DEBUG: 特征" << f << " 有效分裂数=" << validSplits << std::endl;
        }
    }
    
    if (callCount <= 3) {
        std::cout << "DEBUG: 最终结果 - 特征=" << bestFeature 
                  << " 阈值=" << std::fixed << std::setprecision(6) << bestThreshold 
                  << " 增益=" << bestGain << std::endl;
    }
    
    return {bestFeature, bestThreshold, bestGain};
}

std::pair<double, double> XGBoostSplitFinder::computeGradHessStats(
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<int>& indices) const {
    
    double G = 0.0, H = 0.0;
    for (int idx : indices) {
        G += gradients[idx];
        H += hessians[idx];
    }
    return {G, H};
}