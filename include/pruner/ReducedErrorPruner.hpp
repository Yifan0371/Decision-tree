#ifndef REDUCED_ERROR_PRUNER_HPP
#define REDUCED_ERROR_PRUNER_HPP
#include "tree/IPruner.hpp"
#include <vector>

/** Reduced-Error Pruning：若把子树替换成叶子不升高验证误差，则剪掉 */
class ReducedErrorPruner : public IPruner {
public:
    ReducedErrorPruner(const std::vector<double>& X_val, int rowLen,
                       const std::vector<double>& y_val)
        : Xv_(X_val), D_(rowLen), yv_(y_val) {}
    void prune(std::unique_ptr<Node>& root) const override;
private:
    const std::vector<double>& Xv_;
    int D_;
    const std::vector<double>& yv_;
    double validate(Node* node) const;              // 计算节点子树在验证集的 MSE
    void pruneRec(std::unique_ptr<Node>& node) const;
};
#endif