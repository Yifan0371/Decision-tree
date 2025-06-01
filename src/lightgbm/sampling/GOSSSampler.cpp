#include "lightgbm/sampling/GOSSSampler.hpp"
#include <algorithm>
#include <cmath>

void GOSSSampler::sample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    
    // 创建梯度绝对值与索引的配对
    std::vector<std::pair<double, int>> gradWithIndex;
    gradWithIndex.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex.emplace_back(std::abs(gradients[i]), static_cast<int>(i));
    }
    
    // 按梯度绝对值降序排序
    std::sort(gradWithIndex.begin(), gradWithIndex.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算采样数量
    size_t topNum = static_cast<size_t>(n * topRate_);
    size_t randNum = static_cast<size_t>((n - topNum) * otherRate_);
    
    sampleIndices.clear();
    sampleWeights.clear();
    sampleIndices.reserve(topNum + randNum);
    sampleWeights.reserve(topNum + randNum);
    
    // 保留所有大梯度样本
    for (size_t i = 0; i < topNum; ++i) {
        sampleIndices.push_back(gradWithIndex[i].second);
        sampleWeights.push_back(1.0);
    }
    
    // 随机采样小梯度样本
    if (randNum > 0 && n > topNum) {
        std::uniform_int_distribution<size_t> dist(topNum, n - 1);
        std::vector<bool> selected(n, false);
        
        // 标记大梯度样本
        for (size_t i = 0; i < topNum; ++i) {
            selected[gradWithIndex[i].second] = true;
        }
        
        size_t sampledCount = 0;
        while (sampledCount < randNum) {
            size_t idx = dist(gen_);
            if (!selected[gradWithIndex[idx].second]) {
                selected[gradWithIndex[idx].second] = true;
                sampleIndices.push_back(gradWithIndex[idx].second);
                
                // 修复：小梯度样本权重应该是 (1-a)/b，放大其重要性
                double weight = (1.0 - topRate_) / otherRate_;
                sampleWeights.push_back(weight);
                ++sampledCount;
            }
        }
    }
}
