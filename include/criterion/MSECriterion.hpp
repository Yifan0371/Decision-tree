// include/criterion/MSECriterion.hpp
#ifndef MSE_CRITERION_HPP
#define MSE_CRITERION_HPP

#include "../tree/ISplitCriterion.hpp"

class MSECriterion : public ISplitCriterion {
public:
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override;

    // 新增：增量式计算接口，避免重复计算
    struct MetricCache {
        double sum = 0.0;
        double sumSq = 0.0;
        size_t count = 0;
        double mean = 0.0;
        double mse = 0.0;
        bool valid = false;
    };

    // 计算指定索引范围的统计信息
    static void calculateStats(const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              MetricCache& cache);

    // 增量式分割计算
    static double splitMetric(const MetricCache& leftCache,
                             const MetricCache& rightCache);
};

#endif // MSE_CRITERION_HPP