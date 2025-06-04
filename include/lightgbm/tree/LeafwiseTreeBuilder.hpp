// =============================================================================
// include/lightgbm/tree/LeafwiseTreeBuilder.hpp
// 深度 OpenMP 并行优化版本（减少锁竞争、提高阈值、预分配缓冲）
// =============================================================================
#ifndef LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP
#define LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP

#include "tree/Node.hpp"
#include "tree/ISplitFinder.hpp"
#include "tree/ISplitCriterion.hpp"
#include "lightgbm/core/LightGBMConfig.hpp"
#include "lightgbm/sampling/GOSSSampler.hpp"
#include "lightgbm/feature/FeatureBundler.hpp"
#include <queue>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

/** 叶子节点信息（用于优先队列） */
struct LeafInfo {
    Node* node;
    std::vector<int> sampleIndices;
    double splitGain;
    int bestFeature;
    double bestThreshold;
    // Comparator: 按 splitGain 大小排序
    bool operator<(const LeafInfo& other) const {
        return splitGain < other.splitGain;
    }
};

/** Leaf-wise 树构建器 - 深度 OpenMP 并行优化版本 */
class LeafwiseTreeBuilder {
public:
    LeafwiseTreeBuilder(const LightGBMConfig& config,
                        std::unique_ptr<ISplitFinder> finder,
                        std::unique_ptr<ISplitCriterion> criterion)
        : config_(config), finder_(std::move(finder)), criterion_(std::move(criterion)) {
        tempIndices_.reserve(10000);
        leftIndices_.reserve(5000);
        rightIndices_.reserve(5000);
        leftWeights_.reserve(5000);
        rightWeights_.reserve(5000);
    }

    /**
     * 构建单棵树
     * @param data  训练数据
     * @param rowLength 特征数
     * @param labels 原始标签（未使用，仅作兼容）
     * @param targets 残差/梯度
     * @param sampleIndices GOSS 采样的样本索引
     * @param sampleWeights GOSS 采样的权重
     * @param bundles 特征绑定信息
     */
    std::unique_ptr<Node> buildTree(const std::vector<double>& data,
                                    int rowLength,
                                    const std::vector<double>& /* labels */,
                                    const std::vector<double>& targets,
                                    const std::vector<int>& sampleIndices,
                                    const std::vector<double>& sampleWeights,
                                    const std::vector<FeatureBundle>& bundles);

private:
    const LightGBMConfig& config_;
    std::unique_ptr<ISplitFinder> finder_;
    std::unique_ptr<ISplitCriterion> criterion_;

    // Max-heap: 当前待分裂的叶子
    std::priority_queue<LeafInfo> leafQueue_;

    // 预分配内存池
    std::vector<int> tempIndices_;
    std::vector<int> leftIndices_, rightIndices_;
    std::vector<double> leftWeights_, rightWeights_;

    // 单次分裂局部缓冲，避免并行内多次分配
    std::vector<LeafInfo> localNewLeafInfos_;

    // 串行版：保留原有接口
    bool findBestSplitSerial(const std::vector<double>& data,
                             int rowLength,
                             const std::vector<double>& targets,
                             const std::vector<int>& indices,
                             const std::vector<double>& weights,
                             LeafInfo& leafInfo);

    void splitLeafSerial(LeafInfo& leafInfo,
                         const std::vector<double>& data,
                         int rowLength,
                         const std::vector<double>& targets,
                         const std::vector<double>& sampleWeights);

    // 并行版：在需要时调用
    bool findBestSplitParallel(const std::vector<double>& data,
                               int rowLength,
                               const std::vector<double>& targets,
                               const std::vector<int>& indices,
                               const std::vector<double>& weights,
                               LeafInfo& leafInfo);

    void splitLeafParallel(LeafInfo& leafInfo,
                           const std::vector<double>& data,
                           int rowLength,
                           const std::vector<double>& targets,
                           const std::vector<double>& sampleWeights);

    double computeLeafPredictionSerial(const std::vector<int>& indices,
                                       const std::vector<double>& targets,
                                       const std::vector<double>& weights) const;

    double computeLeafPredictionParallel(const std::vector<int>& indices,
                                         const std::vector<double>& targets,
                                         const std::vector<double>& weights) const;

    void processRemainingLeavesSerial(const std::vector<double>& targets,
                                      const std::vector<double>& sampleWeights);

    void processRemainingLeavesParallel(const std::vector<double>& targets,
                                        const std::vector<double>& sampleWeights);
};

#endif // LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP
