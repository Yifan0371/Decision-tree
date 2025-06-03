// =============================================================================
// include/lightgbm/tree/LeafwiseTreeBuilder.hpp - 添加并行方法声明
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

/** 叶子节点信息（用于优先队列） */
struct LeafInfo {
    Node* node;
    std::vector<int> sampleIndices;
    double splitGain;
    int bestFeature;
    double bestThreshold;
    
    // 优先队列比较器（按增益排序）
    bool operator<(const LeafInfo& other) const {
        return splitGain < other.splitGain;
    }
};

/** Leaf-wise树构建器 - 深度OpenMP并行优化版本 */
class LeafwiseTreeBuilder {
public:
    LeafwiseTreeBuilder(const LightGBMConfig& config,
                       std::unique_ptr<ISplitFinder> finder,
                       std::unique_ptr<ISplitCriterion> criterion)
        : config_(config), finder_(std::move(finder)), criterion_(std::move(criterion)) {
        
        // 预分配内存池
        tempIndices_.reserve(10000);
        leftIndices_.reserve(5000);
        rightIndices_.reserve(5000);
    }
    
    /**
     * 构建单棵树
     * @param data 训练数据
     * @param rowLength 特征数
     * @param labels 原始标签
     * @param targets 残差/梯度（训练目标）
     * @param sampleIndices GOSS采样的样本索引
     * @param sampleWeights GOSS采样的权重
     * @param bundles 特征绑定信息
     */
    std::unique_ptr<Node> buildTree(const std::vector<double>& data,
                                   int rowLength,
                                   const std::vector<double>& labels,
                                   const std::vector<double>& targets,
                                   const std::vector<int>& sampleIndices,
                                   const std::vector<double>& sampleWeights,
                                   const std::vector<FeatureBundle>& bundles);

private:
    const LightGBMConfig& config_;
    std::unique_ptr<ISplitFinder> finder_;
    std::unique_ptr<ISplitCriterion> criterion_;
    
    // 内存池（预分配，避免重复分配）
    std::priority_queue<LeafInfo> leafQueue_;
    std::vector<int> tempIndices_;
    std::vector<int> leftIndices_;
    std::vector<int> rightIndices_;
    
    // =============================================
    // 原有方法（保持兼容性）
    // =============================================
    
    bool findBestSplit(const std::vector<double>& data,
                      int rowLength,
                      const std::vector<double>& targets,
                      const std::vector<int>& indices,
                      const std::vector<double>& weights,
                      LeafInfo& leafInfo);
    
    void splitLeaf(LeafInfo& leafInfo,
                  const std::vector<double>& data,
                  int rowLength,
                  const std::vector<double>& targets,
                  const std::vector<double>& sampleWeights);
    
    // =============================================
    // 新增：深度并行优化方法
    // =============================================
    
    /** 并行分裂查找 */
    bool findBestSplitParallel(const std::vector<double>& data,
                              int rowLength,
                              const std::vector<double>& targets,
                              const std::vector<int>& indices,
                              const std::vector<double>& weights,
                              LeafInfo& leafInfo);
    
    /** 并行叶子分裂 */
    void splitLeafParallel(LeafInfo& leafInfo,
                          const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& targets,
                          const std::vector<double>& sampleWeights);
    
    /** 并行叶子预测计算 */
    double computeLeafPredictionParallel(
        const std::vector<int>& indices,
        const std::vector<double>& targets,
        const std::vector<double>& weights) const;
    
    /** 并行处理剩余叶子 */
    void processRemainingLeavesParallel(
        const std::vector<double>& targets,
        const std::vector<double>& sampleWeights);
    
    // =============================================
    // 性能优化工具
    // =============================================
    
    /** 估算节点复杂度（用于并行策略选择） */
    double estimateNodeComplexity(const std::vector<int>& indices) const {
        return static_cast<double>(indices.size()) * config_.maxBin;
    }
    
    /** 获取推荐的并行阈值 */
    size_t getParallelThreshold() const {
        return 1000; // 样本数阈值
    }
    
    /** 检查是否应该使用并行处理 */
    bool shouldUseParallel(size_t sampleCount) const {
        return sampleCount >= getParallelThreshold();
    }
    
    /** 动态负载均衡：按节点大小分配线程 */
    int calculateOptimalThreads(const std::vector<int>& indices) const {
        size_t complexity = indices.size();
        if (complexity < 500) return 1;
        else if (complexity < 2000) return 2;
        else if (complexity < 8000) return 4;
        else return 8; // 最多8线程
    }
};

#endif // LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP