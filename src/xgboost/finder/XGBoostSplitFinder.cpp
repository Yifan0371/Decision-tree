#include "xgboost/finder/XGBoostSplitFinder.hpp"
#include "finder/HistogramEWFinder.hpp"
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<int, double, double> XGBoostSplitFinder::findBestSplit(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& labels,
    const std::vector<int>& indices,
    double currentMetric,
    const ISplitCriterion& criterion) const {
    
    // 使用直方图finder进行快速分裂查找
    static thread_local HistogramEWFinder histFinder(256);
    return histFinder.findBestSplit(data, rowLength, labels, indices, currentMetric, criterion);
}

std::tuple<int, double, double> XGBoostSplitFinder::findBestSplitXGB(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask,
    const std::vector<std::vector<int>>& sortedIndicesAll,
    const XGBoostCriterion& xgbCriterion) const {

    const size_t n = nodeMask.size();

    // 计算父节点统计
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    #pragma omp parallel for reduction(+:G_parent,H_parent,sampleCount) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    if (sampleCount < 2 || H_parent < minChildWeight_) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    constexpr double EPS = 1e-12;

    #pragma omp parallel if(rowLength > 4)
    {
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
        
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < rowLength; ++f) {
            nodeSorted.clear();
            
            const std::vector<int>& featureIndices = sortedIndicesAll[f];
            for (const int idx : featureIndices) {
                if (nodeMask[idx]) {
                    nodeSorted.push_back(idx);
                }
            }
            
            if (nodeSorted.size() < 2) continue;

            double G_left = 0.0, H_left = 0.0;
            for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
                const int idx = nodeSorted[i];
                G_left += gradients[idx];
                H_left += hessians[idx];

                const int nextIdx = nodeSorted[i + 1];
                const double currentVal = data[idx * rowLength + f];
                const double nextVal = data[nextIdx * rowLength + f];

                if (std::abs(nextVal - currentVal) < EPS) continue;

                const double G_right = G_parent - G_left;
                const double H_right = H_parent - H_left;

                if (H_left < minChildWeight_ || H_right < minChildWeight_) continue;

                const double gain = xgbCriterion.computeSplitGain(
                    G_left, H_left, G_right, H_right, G_parent, H_parent, gamma_);

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = 0.5 * (currentVal + nextVal);
                }
            }
        }
        
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}