#ifndef XGBOOST_MODEL_XGBOOSTMODEL_HPP
#define XGBOOST_MODEL_XGBOOSTMODEL_HPP

#include "tree/Node.hpp"
#include <vector>
#include <memory>
#include <algorithm>    

/** XGBoost模型：管理带权重的boosting树集合 */
class XGBoostModel {
public:
    struct XGBTree {
        std::unique_ptr<Node> tree;
        double weight;        // 树权重（通常为学习率）
        double baseScore;     // 该轮的基准分数
        
        XGBTree(std::unique_ptr<Node> t, double w, double base = 0.0)
            : tree(std::move(t)), weight(w), baseScore(base) {}
        
        // 移动语义
        XGBTree(XGBTree&& other) noexcept
            : tree(std::move(other.tree)), weight(other.weight), baseScore(other.baseScore) {}
        
        XGBTree& operator=(XGBTree&& other) noexcept {
            if (this != &other) {
                tree = std::move(other.tree);
                weight = other.weight;
                baseScore = other.baseScore;
            }
            return *this;
        }
    };
    
    XGBoostModel() : globalBaseScore_(0.0) {
        trees_.reserve(200);  // 预分配空间
    }
    
    /** 添加新树到模型 */
    void addTree(std::unique_ptr<Node> tree, double weight = 1.0) {
        trees_.emplace_back(std::move(tree), weight, globalBaseScore_);
    }
    
    /** 单样本预测 */
    double predict(const double* sample, int rowLength) const {
        double prediction = globalBaseScore_;
        for (const auto& xgbTree : trees_) {
            double treePred = predictSingleTree(xgbTree.tree.get(), sample, rowLength);
            prediction += xgbTree.weight * treePred;
        }
        return prediction;
    }
    
    /** 批量预测（内存优化版本） */
    std::vector<double> predictBatch(const std::vector<double>& X, int rowLength) const {
        size_t n = X.size() / rowLength;
        std::vector<double> predictions(n, globalBaseScore_);
        
        // 按树遍历，提高缓存效率
        for (const auto& xgbTree : trees_) {
            for (size_t i = 0; i < n; ++i) {
                const double* sample = &X[i * rowLength];
                double treePred = predictSingleTree(xgbTree.tree.get(), sample, rowLength);
                predictions[i] += xgbTree.weight * treePred;
            }
        }
        return predictions;
    }
    
    /** 获取模型信息 */
    size_t getTreeCount() const { return trees_.size(); }
    void setGlobalBaseScore(double score) { globalBaseScore_ = score; }
    double getGlobalBaseScore() const { return globalBaseScore_; }
    
    /** 获取特征重要性 */
    std::vector<double> getFeatureImportance(int numFeatures) const {
        std::vector<double> importance(numFeatures, 0.0);
        for (const auto& xgbTree : trees_) {
            addTreeImportance(xgbTree.tree.get(), importance);
        }
        
        // 归一化
        double total = 0.0;
        for (double imp : importance) total += imp;
        if (total > 0) {
            for (double& imp : importance) imp /= total;
        }
        
        return importance;
    }
    
    /** 清空模型 */
    void clear() {
        trees_.clear();
        trees_.shrink_to_fit();
        globalBaseScore_ = 0.0;
    }

private:
    std::vector<XGBTree> trees_;
    double globalBaseScore_;
    
    /** 单棵树预测 */
    inline double predictSingleTree(const Node* tree, const double* sample, int rowLength) const {
        const Node* cur = tree;
        while (cur && !cur->isLeaf) {
            double value = sample[cur->getFeatureIndex()];
            cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        return cur ? cur->getPrediction() : 0.0;
    }
    
    /** 累积特征重要性 */
    void addTreeImportance(const Node* node, std::vector<double>& importance) const {
        if (!node || node->isLeaf) return;
        
        int feature = node->getFeatureIndex();
        if (feature >= 0 && feature < static_cast<int>(importance.size())) {
            importance[feature] += 1.0;  // 基于分裂次数的重要性
        }
        
        addTreeImportance(node->getLeft(), importance);
        addTreeImportance(node->getRight(), importance);
    }
};

#endif // XGBOOST_MODEL_XGBOOSTMODEL_HPP