#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

#include <memory>
#include <cstddef>

struct Node {
    bool   isLeaf      = false;
    size_t samples     = 0;
    double metric      = 0.0;      // 节点误差，也用于剪枝
    
    // 使用 union 共享内部节点和叶节点不同字段以节省空间
    union NodeInfo {
        struct InternalNode { 
            int featureIndex;
            double threshold;
            Node* left;
            Node* right;
        } internal;
        
        struct LeafNode {
            double prediction;
            double nodePrediction;  // 用于剪枝的预测值
        } leaf;
        
        // 构造函数
        NodeInfo() {
            // 初始化为内部节点，避免野指针
            internal.featureIndex = -1;
            internal.threshold = 0.0;
            internal.left = nullptr;
            internal.right = nullptr;
        }
        
        // 析构函数（union不会自动调用成员析构）
        ~NodeInfo() {
            // 由Node负责管理内存，这里不需要特殊处理
        }
    } info;
    
    // 用于管理子节点内存的智能指针（不放在union中）
    std::unique_ptr<Node> leftChild  = nullptr;
    std::unique_ptr<Node> rightChild = nullptr;
    
    Node() : isLeaf(false), samples(0), metric(0.0) {
        // union已在其构造函数中初始化
    }
    
    // 设置为叶节点
    void makeLeaf(double prediction, double nodePrediction = 0.0) {
        isLeaf = true;
        info.leaf.prediction = prediction;
        info.leaf.nodePrediction = (nodePrediction != 0.0) ? nodePrediction : prediction;
        // 释放子节点
        leftChild.reset();
        rightChild.reset();
    }
    
    // 设置为内部节点
    void makeInternal(int featureIndex, double threshold) {
        isLeaf = false;
        info.internal.featureIndex = featureIndex;
        info.internal.threshold = threshold;
        info.internal.left = nullptr;
        info.internal.right = nullptr;
    }
    
    // 访问器方法
    int getFeatureIndex() const { 
        return isLeaf ? -1 : info.internal.featureIndex; 
    }
    
    double getThreshold() const { 
        return isLeaf ? 0.0 : info.internal.threshold; 
    }
    
    double getPrediction() const { 
        return isLeaf ? info.leaf.prediction : 0.0; 
    }
    
    double getNodePrediction() const {
        return isLeaf ? info.leaf.nodePrediction : 0.0;
    }
    
    Node* getLeft() const { 
        return isLeaf ? nullptr : leftChild.get(); 
    }
    
    Node* getRight() const { 
        return isLeaf ? nullptr : rightChild.get(); 
    }
    
    // 兼容性访问器（保持原有API）
    int featureIndex() const { return getFeatureIndex(); }
    double threshold() const { return getThreshold(); }
    double prediction() const { return getPrediction(); }
    Node* left() const { return getLeft(); }
    Node* right() const { return getRight(); }
};

#endif // TREE_NODE_HPP