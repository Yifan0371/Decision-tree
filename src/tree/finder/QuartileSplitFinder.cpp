#include "finder/QuartileSplitFinder.hpp"
#include <algorithm>
#include <limits>

static double percentile(std::vector<double> v, double q) { // 修正：复制输入向量
    if (v.empty()) return 0.0;
    size_t k = static_cast<size_t>(q * (v.size() - 1));
    if (k >= v.size()) k = v.size() - 1;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

std::tuple<int,double,double> QuartileSplitFinder::findBestSplit(
    const std::vector<double>& X, int D,
    const std::vector<double>& y,
    const std::vector<int>& idx,
    double parentMetric,
    const ISplitCriterion& crit) const
{
    int bestFeat = -1; 
    double bestThr = 0, bestGain = -std::numeric_limits<double>::infinity();

    for (int f = 0; f < D; ++f) {
        std::vector<double> vals; 
        vals.reserve(idx.size());
        for (int i : idx) vals.push_back(X[i*D+f]);

        if (vals.size() < 4) continue; // 至少需要4个值才能计算四分位数

        double q1 = percentile(vals, 0.25);
        double q2 = percentile(vals, 0.50);
        double q3 = percentile(vals, 0.75);

        // 去重四分位数
        std::vector<double> thresholds;
        thresholds.push_back(q1);
        if (q2 != q1) thresholds.push_back(q2);
        if (q3 != q2 && q3 != q1) thresholds.push_back(q3);

        for (double thr : thresholds) {
            std::vector<int> L, R;
            for (int i : idx) {
                if (X[i*D+f] <= thr) L.push_back(i);
                else R.push_back(i);
            }
            if (L.empty() || R.empty()) continue;

            double mL = crit.nodeMetric(y, L);
            double mR = crit.nodeMetric(y, R);
            double gain = parentMetric - (mL*L.size() + mR*R.size()) / idx.size();
            if (gain > bestGain) {
                bestGain = gain; 
                bestFeat = f; 
                bestThr = thr;
            }
        }
    }
    return {bestFeat, bestThr, bestGain};
}