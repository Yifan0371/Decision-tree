#include "finder/RandomSplitFinder.hpp"
#include <unordered_set>
#include <limits>

std::tuple<int,double,double> RandomSplitFinder::findBestSplit(
    const std::vector<double>& X, int D,
    const std::vector<double>& y,
    const std::vector<int>& idx,
    double parentMetric,
    const ISplitCriterion& crit) const
{
    int bestFeat = -1; 
    double bestThr = 0, bestGain = -std::numeric_limits<double>::infinity();
    std::uniform_real_distribution<> uni01(0.0, 1.0);

    // 对每个特征尝试k个随机阈值
    for (int f = 0; f < D; ++f) {
        // 找到该特征的值域
        double vmin = std::numeric_limits<double>::infinity();
        double vmax = -vmin;
        for (int i : idx) {
            double v = X[i*D+f];
            vmin = std::min(vmin, v); 
            vmax = std::max(vmax, v);
        }
        if (vmax == vmin) continue; // 特征值都相同
        
        // 采样k个随机阈值
        std::unordered_set<double> tried;
        for (int r = 0; r < k_; ++r) {
            double thr = vmin + uni01(gen_) * (vmax - vmin);
            if (!tried.insert(thr).second) continue; // 避免重复

            std::vector<int> L, R;
            L.reserve(idx.size()/2); 
            R.reserve(idx.size()/2);
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