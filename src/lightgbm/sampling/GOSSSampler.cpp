// =============================================================================
// src/lightgbm/sampling/GOSSSampler.cpp
// OpenMP 深度并行优化版本（包含头文件补充与并行归约修正）
// =============================================================================
#include "lightgbm/sampling/GOSSSampler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <chrono>    // 修复：引入 std::chrono
#ifdef _OPENMP
#include <omp.h>
#endif

void GOSSSampler::sample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    // 如果参数无效，直接全量采样
    if (!validateParameters()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        return;
    }
    // 只有样本量足够大时才并行，否则串行
    if (n >= getParallelThreshold()) {
        sampleParallel(gradients, sampleIndices, sampleWeights);
    } else {
        sampleSerial(gradients, sampleIndices, sampleWeights);
    }
}

void GOSSSampler::sampleParallel(const std::vector<double>& gradients,
                                 std::vector<int>& sampleIndices,
                                 std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    // 并行构建 (|grad|, idx) 对
    std::vector<std::pair<double, int>> gradWithIndex(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex[i] = {std::abs(gradients[i]), static_cast<int>(i)};
    }

    // 并行排序：如果编译器支持 C++17 并行算法，可改用 std::sort(std::execution::par, ...)
    // 这里依旧使用串行 std::sort，但仅在 n 较大时执行
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

    // 并行收集大梯度样本索引
    std::vector<int> topIndices(topNum);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < topNum; ++i) {
        topIndices[i] = gradWithIndex[i].second;
    }
    sampleIndices.insert(sampleIndices.end(), topIndices.begin(), topIndices.end());
    sampleWeights.insert(sampleWeights.end(), topNum, 1.0);

    // 随机采样小梯度样本（串行以保证随机性）
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

    // 如果采样结果为空，则退回全量
    if (sampleIndices.empty()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}

void GOSSSampler::sampleSerial(const std::vector<double>& gradients,
                               std::vector<int>& sampleIndices,
                               std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    std::vector<std::pair<double, int>> gradWithIndex;
    gradWithIndex.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex.emplace_back(std::abs(gradients[i]), static_cast<int>(i));
    }
    std::sort(gradWithIndex.begin(), gradWithIndex.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    size_t topNum = static_cast<size_t>(std::floor(n * topRate_));
    size_t smallGradNum = n - topNum;
    size_t randNum = static_cast<size_t>(std::floor(smallGradNum * otherRate_));
    topNum = std::min(topNum, n);
    randNum = std::min(randNum, smallGradNum);

    sampleIndices.clear();
    sampleWeights.clear();
    sampleIndices.reserve(topNum + randNum);
    sampleWeights.reserve(topNum + randNum);

    // 保留大梯度
    for (size_t i = 0; i < topNum; ++i) {
        sampleIndices.push_back(gradWithIndex[i].second);
        sampleWeights.push_back(1.0);
    }
    // 随机采样小梯度
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
    if (sampleIndices.empty()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}

void GOSSSampler::sampleWithTiming(const std::vector<double>& gradients,
                                   std::vector<int>& sampleIndices,
                                   std::vector<double>& sampleWeights,
                                   double& samplingTimeMs) const {
    auto start = std::chrono::high_resolution_clock::now();
    sample(gradients, sampleIndices, sampleWeights);
    auto end = std::chrono::high_resolution_clock::now();
    samplingTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void GOSSSampler::adaptiveSample(const std::vector<double>& gradients,
                                 std::vector<int>& sampleIndices,
                                 std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    // 先计算均值和标准差，阈值 n < 10000 串行
    double meanGrad = 0.0, stdGrad = 0.0;
    if (n >= 10000) {
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
    } else {
        for (size_t i = 0; i < n; ++i) {
            meanGrad += std::abs(gradients[i]);
        }
        meanGrad /= n;
        for (size_t i = 0; i < n; ++i) {
            double diff = std::abs(gradients[i]) - meanGrad;
            stdGrad += diff * diff;
        }
    }
    stdGrad = std::sqrt(stdGrad / n);

    double adaptiveTopRate = topRate_;
    double adaptiveOtherRate = otherRate_;
    double cv = (meanGrad > 0) ? stdGrad / meanGrad : 1.0;
    if (cv > 2.0) {
        adaptiveTopRate = std::min(0.5, topRate_ * 1.5);
        adaptiveOtherRate = std::max(0.05, otherRate_ * 0.8);
    } else if (cv < 0.5) {
        adaptiveTopRate = std::max(0.1, topRate_ * 0.8);
        adaptiveOtherRate = std::min(0.3, otherRate_ * 1.2);
    }
    GOSSSampler adaptiveSampler(adaptiveTopRate, adaptiveOtherRate);
    adaptiveSampler.sample(gradients, sampleIndices, sampleWeights);
}

GOSSSampler::SamplingStats GOSSSampler::getSamplingStats(
    const std::vector<double>& gradients,
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights) const {
    SamplingStats stats;
    stats.totalSamples = gradients.size();
    stats.selectedSamples = sampleIndices.size();
    stats.samplingRatio = static_cast<double>(stats.selectedSamples) / stats.totalSamples;

    // 先初始化
    stats.effectiveWeightSum = 0.0;
    stats.maxGradient = 0.0;
    stats.minGradient = std::numeric_limits<double>::max();

    size_t m = sampleIndices.size();
    if (m == 0) {
        stats.effectiveWeightSum = 0.0;
        stats.maxGradient = 0.0;
        stats.minGradient = 0.0;
        return stats;
    }

    // 并行计算有效权重和
    double localWeightSum = 0.0;
    if (m >= 2000) {
        #pragma omp parallel for reduction(+:localWeightSum) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            localWeightSum += sampleWeights[i];
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            localWeightSum += sampleWeights[i];
        }
    }
    stats.effectiveWeightSum = localWeightSum;

    // 并行计算最大梯度
    double localMaxGrad = 0.0;
    if (m >= 2000) {
        #pragma omp parallel for reduction(max:localMaxGrad) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad > localMaxGrad) {
                localMaxGrad = grad;
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad > localMaxGrad) {
                localMaxGrad = grad;
            }
        }
    }
    stats.maxGradient = localMaxGrad;

    // 并行计算最小梯度
    double localMinGrad = std::numeric_limits<double>::max();
    if (m >= 2000) {
        #pragma omp parallel for reduction(min:localMinGrad) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad < localMinGrad) {
                localMinGrad = grad;
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad < localMinGrad) {
                localMinGrad = grad;
            }
        }
    }
    stats.minGradient = localMinGrad;

    return stats;
}
