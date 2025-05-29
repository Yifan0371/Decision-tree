#include "finder/ExhaustiveSplitFinder.hpp"
#include <algorithm>
#include <tuple>

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplit(const std::vector<double>& data,
                                     int rowLength,
                                     const std::vector<double>& labels,
                                     const std::vector<int>& indices,
                                     double currentMetric,
                                     const ISplitCriterion& criterion) const {
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestImp  = 0.0;
    size_t nFeat    = rowLength;

    for (size_t f = 0; f < nFeat; ++f) {
        // 把 (value, idx) 排序
        std::vector<std::pair<double, int>> pairs;
        pairs.reserve(indices.size());
        for (int idx : indices) {
            pairs.emplace_back(data[idx * rowLength + f], idx);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });

        // 构造左右索引
        std::vector<int> leftIdx, rightIdx(indices.begin(), indices.end());
        for (size_t i = 0; i + 1 < pairs.size(); ++i) {
            int idx = pairs[i].second;
            leftIdx.push_back(idx);
            rightIdx.erase(std::find(rightIdx.begin(), rightIdx.end(), idx));

            double v1 = pairs[i].first;
            double v2 = pairs[i+1].first;
            if (v1 == v2) continue;

            double thr = 0.5 * (v1 + v2);
            double mLeft  = criterion.nodeMetric(labels, leftIdx);
            double mRight = criterion.nodeMetric(labels, rightIdx);
            double weighted = (mLeft * leftIdx.size() +
                               mRight * rightIdx.size()) / indices.size();
            double imp = currentMetric - weighted;
            if (imp > bestImp) {
                bestImp  = imp;
                bestFeat = f;
                bestThr  = thr;
            }
        }
    }
    return {bestFeat, bestThr, bestImp};
}
