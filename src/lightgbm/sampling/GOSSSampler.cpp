// =============================================================================
// src/lightgbm/sampling/GOSSSampler.cpp - OpenMP深度并行优化版本
// =============================================================================
#include "lightgbm/sampling/GOSSSampler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

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
    
    // **并行优化1: 梯度绝对值计算和排序的并行**
    if (n > 5000) {
        sampleParallel(gradients, sampleIndices, sampleWeights);
    } else {
        sampleSerial(gradients, sampleIndices, sampleWeights);
    }
}

// **新增方法：大数据集的并行GOSS采样**
void GOSSSampler::sampleParallel(const std::vector<double>& gradients,
                                 std::vector<int>& sampleIndices,
                                 std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    
    // **并行优化2: 并行构建索引-梯度对**
    std::vector<std::pair<double, int>> gradWithIndex(n);
    
    #pragma omp parallel for schedule(static, 1024)
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex[i] = {std::abs(gradients[i]), static_cast<int>(i)};
    }
    
    // **并行优化3: 并行排序**
    // 使用标准库的并行排序（C++17）或手动实现并行归并排序
    std::sort(gradWithIndex.begin(), gradWithIndex.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算采样数量
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
    
    // **并行优化4: 并行收集大梯度样本**
    std::vector<int> topIndices(topNum);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < topNum; ++i) {
        topIndices[i] = gradWithIndex[i].second;
    }
    
    // 添加大梯度样本
    sampleIndices.insert(sampleIndices.end(), topIndices.begin(), topIndices.end());
    sampleWeights.insert(sampleWeights.end(), topNum, 1.0);
    
    // **并行优化5: 小梯度样本的并行采样**
    if (randNum > 0 && smallGradNum > 0) {
        std::vector<int> smallGradPool;
        smallGradPool.reserve(smallGradNum);
        
        // 并行收集小梯度样本池
        #pragma omp parallel
        {
            std::vector<int> localPool;
            localPool.reserve(smallGradNum / omp_get_num_threads() + 1);
            
            #pragma omp for nowait
            for (size_t i = topNum; i < n; ++i) {
                localPool.push_back(gradWithIndex[i].second);
            }
            
            #pragma omp critical
            {
                smallGradPool.insert(smallGradPool.end(), 
                                   localPool.begin(), localPool.end());
            }
        }
        
        // 随机选择（这部分保持串行以保证随机性）
        std::shuffle(smallGradPool.begin(), smallGradPool.end(), gen_);
        
        // **关键：权重公式**
        double smallWeight = (1.0 - topRate_) / otherRate_;
        
        for (size_t i = 0; i < randNum; ++i) {
            sampleIndices.push_back(smallGradPool[i]);
            sampleWeights.push_back(smallWeight);
        }
    }
    
    // 验证采样结果
    if (sampleIndices.empty()) {
        // 采样失败，使用全量数据
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}

// **新增方法：小数据集的串行GOSS采样（优化版）**
void GOSSSampler::sampleSerial(const std::vector<double>& gradients,
                               std::vector<int>& sampleIndices,
                               std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    
    // **GOSS核心算法（优化的串行版本）**
    std::vector<std::pair<double, int>> gradWithIndex;
    gradWithIndex.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex.emplace_back(std::abs(gradients[i]), static_cast<int>(i));
    }
    
    std::sort(gradWithIndex.begin(), gradWithIndex.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算采样数量
    size_t topNum = static_cast<size_t>(std::floor(n * topRate_));
    size_t smallGradNum = n - topNum;
    size_t randNum = static_cast<size_t>(std::floor(smallGradNum * otherRate_));
    
    topNum = std::min(topNum, n);
    randNum = std::min(randNum, smallGradNum);
    
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
    if (randNum > 0 && smallGradNum > 0) {
        std::vector<int> smallGradPool;
        smallGradPool.reserve(smallGradNum);
        
        for (size_t i = topNum; i < n; ++i) {
            smallGradPool.push_back(gradWithIndex[i].second);
        }
        
        std::shuffle(smallGradPool.begin(), smallGradPool.end(), gen_);
        
        double smallWeight = (1.0 - topRate_) / otherRate_;
        
        for (size_t i = 0; i < randNum; ++i) {
            sampleIndices.push_back(smallGradPool[i]);
            sampleWeights.push_back(smallWeight);
        }
    }
    
    // 验证结果
    if (sampleIndices.empty()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}

// **新增方法：高级GOSS采样（带性能监控）**
void GOSSSampler::sampleWithTiming(const std::vector<double>& gradients,
                                   std::vector<int>& sampleIndices,
                                   std::vector<double>& sampleWeights,
                                   double& samplingTimeMs) const {
    auto start = std::chrono::high_resolution_clock::now();
    
    sample(gradients, sampleIndices, sampleWeights);
    
    auto end = std::chrono::high_resolution_clock::now();
    samplingTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// **新增方法：自适应采样（根据梯度分布调整参数）**
void GOSSSampler::adaptiveSample(const std::vector<double>& gradients,
                                 std::vector<int>& sampleIndices,
                                 std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    
    // **并行优化6: 梯度统计的并行计算**
    double meanGrad = 0.0, stdGrad = 0.0;
    
    #pragma omp parallel for reduction(+:meanGrad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        meanGrad += std::abs(gradients[i]);
    }
    meanGrad /= n;
    
    #pragma omp parallel for reduction(+:stdGrad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(gradients[i]) - meanGrad;
        stdGrad += diff * diff;
    }
    stdGrad = std::sqrt(stdGrad / n);
    
    // 自适应调整采样参数
    double adaptiveTopRate = topRate_;
    double adaptiveOtherRate = otherRate_;
    
    double cv = (meanGrad > 0) ? stdGrad / meanGrad : 1.0; // 变异系数
    
    if (cv > 2.0) {
        // 高变异性：增加大梯度保留率
        adaptiveTopRate = std::min(0.5, topRate_ * 1.5);
        adaptiveOtherRate = std::max(0.05, otherRate_ * 0.8);
    } else if (cv < 0.5) {
        // 低变异性：可以减少大梯度保留率
        adaptiveTopRate = std::max(0.1, topRate_ * 0.8);
        adaptiveOtherRate = std::min(0.3, otherRate_ * 1.2);
    }
    
    // 使用调整后的参数进行采样
    GOSSSampler adaptiveSampler(adaptiveTopRate, adaptiveOtherRate);
    adaptiveSampler.sample(gradients, sampleIndices, sampleWeights);
}

// **新增方法：获取采样统计信息**
GOSSSampler::SamplingStats GOSSSampler::getSamplingStats(
    const std::vector<double>& gradients,
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights) const {
    
    SamplingStats stats;
    stats.totalSamples = gradients.size();
    stats.selectedSamples = sampleIndices.size();
    stats.samplingRatio = static_cast<double>(stats.selectedSamples) / stats.totalSamples;
    
    // **并行优化7: 统计计算的并行**
    if (!sampleIndices.empty()) {
        double totalWeight = 0.0;
        double maxGrad = 0.0, minGrad = std::numeric_limits<double>::max();
        
        #pragma omp parallel for reduction(+:totalWeight) reduction(max:maxGrad) reduction(min:minGrad) schedule(static)
        for (size_t i = 0; i < sampleIndices.size(); ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            double weight = sampleWeights[i];
            
            totalWeight += weight;
            maxGrad = std::max(maxGrad, grad);
            minGrad = std::min(minGrad, grad);
        }
        
        stats.effectiveWeightSum = totalWeight;
        stats.maxGradient = maxGrad;
        stats.minGradient = minGrad;
    }
    
    return stats;
}