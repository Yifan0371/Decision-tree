#pragma once

#include "tree/Node.hpp"
#include <vector>
#include <memory>


class LightGBMModel {
public:
    struct LGBTree {
        std::unique_ptr<Node> tree;
        double weight;

        LGBTree(std::unique_ptr<Node> t, double w)
            : tree(std::move(t)), weight(w) {}

        LGBTree(LGBTree&& other) noexcept
            : tree(std::move(other.tree)), weight(other.weight) {}
    };

    LightGBMModel() : baseScore_(0.0) {
        trees_.reserve(200);
    }

    void addTree(std::unique_ptr<Node> tree, double weight = 1.0) {
        trees_.emplace_back(std::move(tree), weight);
    }

    double predict(const double* sample, int rowLength) const {
        double prediction = baseScore_;
        for (const auto& lgbTree : trees_) {
            prediction += lgbTree.weight * predictSingleTree(lgbTree.tree.get(), sample, rowLength);
        }
        return prediction;
    }

    std::vector<double> predictBatch(const std::vector<double>& X, int rowLength) const {
        size_t n = X.size() / rowLength;
        std::vector<double> predictions(n, baseScore_);

        for (const auto& lgbTree : trees_) {
            for (size_t i = 0; i < n; ++i) {
                const double* sample = &X[i * rowLength];
                predictions[i] += lgbTree.weight * predictSingleTree(lgbTree.tree.get(), sample, rowLength);
            }
        }
        return predictions;
    }

    size_t getTreeCount() const { return trees_.size(); }
    void setBaseScore(double score) { baseScore_ = score; }
    double getBaseScore() const { return baseScore_; }

    void clear() {
        trees_.clear();
        trees_.shrink_to_fit();
        baseScore_ = 0.0;
    }

    
    std::vector<double> getFeatureImportance(int numFeatures) const {
        if (numFeatures <= 0) {
            return std::vector<double>();
        }
        return std::vector<double>(static_cast<size_t>(numFeatures), 0.0);
    }

private:
    std::vector<LGBTree> trees_;
    double baseScore_;

    inline double predictSingleTree(const Node* tree, const double* sample, int ) const {
        const Node* cur = tree;
        while (cur && !cur->isLeaf) {
            double value = sample[cur->getFeatureIndex()];
            cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        return cur ? cur->getPrediction() : 0.0;
    }
};
