#include "lightgbm/feature/FeatureBundler.hpp"
#include <algorithm>
#include <unordered_map>
#include <queue>

void FeatureBundler::createBundles(const std::vector<double>& data,
                                  int rowLength,
                                  size_t sampleSize,
                                  std::vector<FeatureBundle>& bundles) const {
    bundles.clear();
    
    // 构建特征冲突矩阵
    std::vector<std::vector<double>> conflictMatrix(rowLength, 
                                                   std::vector<double>(rowLength, 0.0));
    buildConflictGraph(data, rowLength, sampleSize, conflictMatrix);
    
    // 计算每个特征的冲突度（用于排序）
    std::vector<std::pair<double, int>> featureConflicts;
    featureConflicts.reserve(rowLength);
    for (int i = 0; i < rowLength; ++i) {
        double totalConflict = 0.0;
        for (int j = 0; j < rowLength; ++j) {
            if (i != j) totalConflict += conflictMatrix[i][j];
        }
        featureConflicts.emplace_back(totalConflict, i);
    }
    
    // 按冲突度降序排序
    std::sort(featureConflicts.begin(), featureConflicts.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 贪心算法进行特征绑定
    std::vector<bool> used(rowLength, false);
    
    for (const auto& [conflict, feature] : featureConflicts) {
        if (used[feature]) continue;
        
        // 创建新的bundle
        FeatureBundle bundle;
        bundle.features.push_back(feature);
        bundle.offsets.push_back(0.0);
        used[feature] = true;
        
        double currentOffset = maxBin_;
        
        // 尝试添加更多特征到当前bundle
        for (int j = 0; j < rowLength; ++j) {
            if (used[j] || conflictMatrix[feature][j] > maxConflictRate_) continue;
            
            // 检查与bundle中所有特征的冲突
            bool canBundle = true;
            for (int bundledFeature : bundle.features) {
                if (conflictMatrix[j][bundledFeature] > maxConflictRate_) {
                    canBundle = false;
                    break;
                }
            }
            
            if (canBundle && currentOffset + maxBin_ <= 65536) { // 避免溢出
                bundle.features.push_back(j);
                bundle.offsets.push_back(currentOffset);
                used[j] = true;
                currentOffset += maxBin_;
            }
        }
        
        bundle.totalBins = static_cast<int>(currentOffset);
        bundles.push_back(std::move(bundle));
    }
}

double FeatureBundler::calculateConflictRate(const std::vector<double>& data,
                                           int rowLength, size_t sampleSize,
                                           int feat1, int feat2) const {
    const double EPS = 1e-12;
    size_t conflicts = 0;
    
    for (size_t i = 0; i < sampleSize; ++i) {
        double val1 = data[i * rowLength + feat1];
        double val2 = data[i * rowLength + feat2];
        
        // 如果两个特征都非零，则认为是冲突
        if (std::abs(val1) > EPS && std::abs(val2) > EPS) {
            ++conflicts;
        }
    }
    
    return static_cast<double>(conflicts) / sampleSize;
}

void FeatureBundler::buildConflictGraph(const std::vector<double>& data,
                                      int rowLength, size_t sampleSize,
                                      std::vector<std::vector<double>>& conflictMatrix) const {
    // 并行化构建冲突矩阵（这里简化为串行）
    for (int i = 0; i < rowLength; ++i) {
        for (int j = i + 1; j < rowLength; ++j) {
            double conflict = calculateConflictRate(data, rowLength, sampleSize, i, j);
            conflictMatrix[i][j] = conflictMatrix[j][i] = conflict;
        }
    }
}

std::pair<int, double> FeatureBundler::transformFeature(int originalFeature,
                                                       double value,
                                                       const std::vector<FeatureBundle>& bundles) const {
    // 找到特征所属的bundle
    for (size_t bundleIdx = 0; bundleIdx < bundles.size(); ++bundleIdx) {
        const auto& bundle = bundles[bundleIdx];
        auto it = std::find(bundle.features.begin(), bundle.features.end(), originalFeature);
        if (it != bundle.features.end()) {
            size_t pos = std::distance(bundle.features.begin(), it);
            double transformedValue = value + bundle.offsets[pos];
            return {static_cast<int>(bundleIdx), transformedValue};
        }
    }
    
    // 如果没找到，返回原值（这种情况不应该发生）
    return {originalFeature, value};
}
