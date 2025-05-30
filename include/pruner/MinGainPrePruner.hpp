#ifndef MIN_GAIN_PRE_PRUNER_HPP
#define MIN_GAIN_PRE_PRUNER_HPP
#include "tree/IPruner.hpp"

/** 预剪枝：当 split gain < minGain 时直接终止分裂 */
class MinGainPrePruner : public IPruner {
public:
    explicit MinGainPrePruner(double minGain) : minGain_(minGain) {}
    /** 对预剪枝，接口只是存参数，在训练时由 Trainer 主动查询 */
    void prune(std::unique_ptr<Node>&) const override {}   // 空实现
    double minGain() const { return minGain_; }
private:
    double minGain_;
};
#endif