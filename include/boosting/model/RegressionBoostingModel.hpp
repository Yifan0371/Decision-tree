#pragma once

#include "tree/Node.hpp"
#include <vector>
#include <memory>


class RegressionBoostingModel {
public:
    struct RegressionTree {
        std::unique_ptr<Node> tree;
        double weight;
        double learningRate;
        
        RegressionTree(std::unique_ptr<Node> t, double w, double lr)
            : tree(std::move(t)), weight(w), learningRate(lr) {}
        
        
        RegressionTree(RegressionTree&& other) noexcept
            : tree(std::move(other.tree)), weight(other.weight), learningRate(other.learningRate) {}
        
        RegressionTree& operator=(RegressionTree&& other) noexcept {
            if (this != &other) {
                tree = std::move(other.tree);
                weight = other.weight;
                learningRate = other.learningRate;
            }
            return *this;
        }
    };
    
    RegressionBoostingModel() : baseScore_(0.0) {
        trees_.reserve(100);  
    }
    
    
    void addTree(std::unique_ptr<Node> tree, double weight = 1.0, double learningRate = 1.0) {
        trees_.emplace_back(std::move(tree), weight, learningRate);
    }
    
    
    double predict(const double* sample, int rowLength) const {
        double prediction = baseScore_;
        for (const auto& regTree : trees_) {
            double treePred = predictSingleTree(regTree.tree.get(), sample, rowLength);
            prediction += regTree.learningRate * regTree.weight * treePred;
        }
        return prediction;
    }
    
    
    std::vector<double> predictBatch(const std::vector<double>& X, int rowLength) const {
        size_t n = X.size() / rowLength;
        std::vector<double> predictions(n, baseScore_);
        
        
        for (const auto& regTree : trees_) {
            double factor = regTree.learningRate * regTree.weight;
            for (size_t i = 0; i < n; ++i) {
                const double* sample = &X[i * rowLength];
                double treePred = predictSingleTree(regTree.tree.get(), sample, rowLength);
                predictions[i] += factor * treePred;
            }
        }
        return predictions;
    }
    
    
    size_t getTreeCount() const { return trees_.size(); }
    
    
    void setBaseScore(double score) { baseScore_ = score; }
    double getBaseScore() const { return baseScore_; }
    
    
    void getModelStats(int& totalDepth, int& totalLeaves, size_t& memoryUsage) const {
        totalDepth = 0;
        totalLeaves = 0;
        memoryUsage = 0;
        
        for (const auto& regTree : trees_) {
            int depth = 0, leaves = 0;
            calculateTreeStats(regTree.tree.get(), 0, depth, leaves);
            totalDepth += depth;
            totalLeaves += leaves;
            memoryUsage += estimateTreeMemory(regTree.tree.get());
        }
    }
    
    const std::vector<RegressionTree>& getTrees() const { return trees_; }
    std::vector<RegressionTree>& getTrees() { return trees_; }
    
    std::vector<double> getFeatureImportance(int numFeatures) const {
        std::vector<double> importance(numFeatures, 0.0);
        for (const auto& regTree : trees_) {
            addTreeImportance(regTree.tree.get(), importance);
        }
        
        
        double total = 0.0;
        for (double imp : importance) total += imp;
        if (total > 0) {
            for (double& imp : importance) imp /= total;
        }
        
        return importance;
    }
    
    
    void clear() {
        trees_.clear();
        trees_.shrink_to_fit();
        baseScore_ = 0.0;
    }

private:
    std::vector<RegressionTree> trees_;
    double baseScore_;
    
   
    inline double predictSingleTree(const Node* tree, const double* sample, int ) const {
        const Node* cur = tree;
        while (cur && !cur->isLeaf) {
            double value = sample[cur->getFeatureIndex()];
            cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        return cur ? cur->getPrediction() : 0.0;
    }
    
    
    void calculateTreeStats(const Node* node, int currentDepth, 
                           int& maxDepth, int& leafCount) const {
        if (!node) return;
        
        maxDepth = std::max(maxDepth, currentDepth);
        
        if (node->isLeaf) {
            leafCount++;
        } else {
            calculateTreeStats(node->getLeft(), currentDepth + 1, maxDepth, leafCount);
            calculateTreeStats(node->getRight(), currentDepth + 1, maxDepth, leafCount);
        }
    }
    
    
    size_t estimateTreeMemory(const Node* node) const {
        if (!node) return 0;
        size_t size = sizeof(Node);
        if (!node->isLeaf) {
            size += estimateTreeMemory(node->getLeft());
            size += estimateTreeMemory(node->getRight());
        }
        return size;
    }
    
    
    void addTreeImportance(const Node* node, std::vector<double>& importance) const {
        if (!node || node->isLeaf) return;
        
        int feature = node->getFeatureIndex();
        if (feature >= 0 && feature < static_cast<int>(importance.size())) {
            
            importance[feature] += node->samples;
        }
        
        addTreeImportance(node->getLeft(), importance);
        addTreeImportance(node->getRight(), importance);
    }
};
