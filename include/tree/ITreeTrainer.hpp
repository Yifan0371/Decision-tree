#pragma once

#include <vector>
#include "Node.hpp"

class ITreeTrainer {
public:
    virtual ~ITreeTrainer() = default;

    virtual void train(const std::vector<double>& data,
                       int rowLength,
                       const std::vector<double>& labels) = 0;

    virtual double predict(const double* sample,
                           int rowLength) const = 0;

    virtual void evaluate(const std::vector<double>& X,
                          int rowLength,
                          const std::vector<double>& y,
                          double& mse,
                          double& mae) = 0;

    const Node* getRoot() const { return root_.get(); }

protected:
    std::unique_ptr<Node> root_;
};
