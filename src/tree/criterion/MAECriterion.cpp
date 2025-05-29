#include "criterion/MAECriterion.hpp"
#include <algorithm>
#include <cmath>

/* --------- 工具：子集中位数 ---------- */
static double subsetMedian(const std::vector<double>& y,
                           const std::vector<int>& idx)
{
    std::vector<double> v;
    v.reserve(idx.size());
    for (int i : idx) v.push_back(y[i]);

    const size_t n = v.size();
    auto mid_it = v.begin() + n / 2;
    std::nth_element(v.begin(), mid_it, v.end());

    if (n % 2 == 1)                // 奇数
        return *mid_it;
    else {                         // 偶数：再找左中位
        auto left_it = std::max_element(v.begin(), mid_it);
        return 0.5 * (*left_it + *mid_it);
    }
}

/* --------- MAE 计算 ---------- */
double MAECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const
{
    if (indices.empty()) return 0.0;

    const double med = subsetMedian(labels, indices);
    double sumAbs = 0.0;
    for (int i : indices) sumAbs += std::abs(labels[i] - med);
    return sumAbs / indices.size();
}
