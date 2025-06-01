#ifndef LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP
#define LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP

#include "tree/Node.hpp"
#include "tree/ISplitFinder.hpp"
#include "tree/ISplitCriterion.hpp"
#include "lightgbm/core/LightGBMConfig.hpp"  // 添加这行
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

/** Leaf-wise树构建器 */
class LeafwiseTreeBuilder {
public:
    LeafwiseTreeBuilder(const LightGBMConfig& config,
                       std::unique_ptr<ISplitFinder> finder,
                       std::unique_ptr<ISplitCriterion> criterion)
        : config_(config), finder_(std::move(finder)), criterion_(std::move(criterion)) {
        
        // 预分配内存池 - 移除 leafQueue_.reserve() 因为priority_queue没有此方法
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
                                   const std::vector<double>& targets,  // 新增参数
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
    
    bool findBestSplit(const std::vector<double>& data,
                      int rowLength,
                      const std::vector<double>& targets,  // 修改参数名
                      const std::vector<int>& indices,
                      const std::vector<double>& weights,
                      LeafInfo& leafInfo);
    
    void splitLeaf(LeafInfo& leafInfo,
                  const std::vector<double>& data,
                  int rowLength,
                  const std::vector<double>& targets,     // 修改参数
                  const std::vector<double>& sampleWeights); // 新增参数
};

#endif // LIGHTGBM_TREE_LEAFWISETREEBUILDER_HPP
