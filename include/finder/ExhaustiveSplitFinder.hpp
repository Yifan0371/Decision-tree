// include/finder/ExhaustiveSplitFinder.hpp
#ifndef EXHAUSTIVE_SPLIT_FINDER_HPP
#define EXHAUSTIVE_SPLIT_FINDER_HPP

#include "../tree/ISplitFinder.hpp"
#include "../criterion/MSECriterion.hpp"

class ExhaustiveSplitFinder : public ISplitFinder {
public:
    std::tuple<int, double, double>
    findBestSplit(const std::vector<double>& data,
                  int rowLength,
                  const std::vector<double>& labels,
                  const std::vector<int>& indices,
                  double currentMetric,
                  const ISplitCriterion& criterion) const override;

private:
    // 高效的增量分割计算
    struct SplitCandidate {
        double value;
        int index;
        bool operator<(const SplitCandidate& other) const {
            return value < other.value;
        }
    };

    // 针对 MSE 的特化优化
    std::tuple<int, double, double>
    findBestSplitMSE(const std::vector<double>& data,
                     int rowLength,
                     const std::vector<double>& labels,
                     const std::vector<int>& indices,
                     double currentMetric) const;

    // 通用分割查找
    std::tuple<int, double, double>
    findBestSplitGeneric(const std::vector<double>& data,
                        int rowLength,
                        const std::vector<double>& labels,
                        const std::vector<int>& indices,
                        double currentMetric,
                        const ISplitCriterion& criterion) const;
};

#endif // EXHAUSTIVE_SPLIT_FINDER_HPP