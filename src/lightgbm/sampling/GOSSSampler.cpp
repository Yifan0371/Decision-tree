#include "lightgbm/sampling/GOSSSampler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

void GOSSSampler::sample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    
    // 参数验证
    if (topRate_ <= 0.0 || topRate_ >= 1.0 || otherRate_ <= 0.0 || otherRate_ >= 1.0) {
        // 无效参数，使用全量采样
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        return;
    }
    
    // **GOSS核心算法**
    // 步骤1: 按梯度绝对值排序
    std::vector<std::pair<double, int>> gradWithIndex;
    gradWithIndex.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex.emplace_back(std::abs(gradients[i]), static_cast<int>(i));
    }
    
    std::sort(gradWithIndex.begin(), gradWithIndex.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 步骤2: 计算采样数量
    size_t topNum = static_cast<size_t>(std::floor(n * topRate_));
    size_t smallGradNum = n - topNum;
    size_t randNum = static_cast<size_t>(std::floor(smallGradNum * otherRate_));
    
    // 边界检查
    topNum = std::min(topNum, n);
    randNum = std::min(randNum, smallGradNum);
    
    sampleIndices.clear();
    sampleWeights.clear();
    sampleIndices.reserve(topNum + randNum);
    sampleWeights.reserve(topNum + randNum);
    
    // 步骤3: 保留所有大梯度样本（权重=1）
    for (size_t i = 0; i < topNum; ++i) {
        sampleIndices.push_back(gradWithIndex[i].second);
        sampleWeights.push_back(1.0);
    }
    
    // 步骤4: 随机采样小梯度样本
    if (randNum > 0 && smallGradNum > 0) {
        // 创建小梯度样本池
        std::vector<int> smallGradPool;
        smallGradPool.reserve(smallGradNum);
        for (size_t i = topNum; i < n; ++i) {
            smallGradPool.push_back(gradWithIndex[i].second);
        }
        
        // 随机选择
        std::shuffle(smallGradPool.begin(), smallGradPool.end(), gen_);
        
        // **关键：论文权重公式**
        // 小梯度样本权重 = (1-a)/b，其中a=topRate_, b=otherRate_
        double smallWeight = (1.0 - topRate_) / otherRate_;
        
        for (size_t i = 0; i < randNum; ++i) {
            sampleIndices.push_back(smallGradPool[i]);
            sampleWeights.push_back(smallWeight);
        }
    }
    
    // **验证：确保采样合理性**
    if (sampleIndices.empty()) {
        // 采样失败，使用全量数据
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}