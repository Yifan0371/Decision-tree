// =============================================================================
// include/boosting/loss/IRegressionLoss.hpp - 添加并行方法声明
// =============================================================================
#ifndef BOOSTING_LOSS_IREGRESSIONLOSS_HPP
#define BOOSTING_LOSS_IREGRESSIONLOSS_HPP

#include <vector>
#include <string>
#include <chrono>

/**
 * 回归损失函数接口：专门为回归任务设计
 * 支持一阶和二阶梯度计算，为XGBoost等高级算法预留接口
 * 新增：OpenMP并行优化支持
 */
class IRegressionLoss {
public:
    virtual ~IRegressionLoss() = default;
    
    /** 计算单个样本的损失值 */
    virtual double loss(double y_true, double y_pred) const = 0;
    
    /** 计算一阶梯度（负梯度作为伪残差） */
    virtual double gradient(double y_true, double y_pred) const = 0;
    
    /** 计算二阶梯度（Hessian对角元素，为XGBoost预留） */
    virtual double hessian(double y_true, double y_pred) const = 0;
    
    /** 
     * 批量计算梯度和Hessian（内存优化版本）
     * 这是性能关键路径，已进行OpenMP并行优化
     */
    virtual void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const;
    
    /** 获取损失函数名称 */
    virtual std::string name() const = 0;
    
    /** 是否支持二阶优化（XGBoost需要） */
    virtual bool supportsSecondOrder() const { return false; }
    
    // =============================================
    // 新增：并行优化的批量计算方法
    // =============================================
    
    /** 
     * 批量损失计算（并行优化版本）
     * 专门用于训练过程中的损失监控
     */
    virtual double computeBatchLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const;
    
    /** 
     * 仅计算梯度的并行版本（GBDT常用）
     * 比computeGradientsHessians更轻量，适用于不需要二阶导数的算法
     */
    virtual void computeBatchGradients(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients) const;
    
    /** 
     * SIMD友好的向量化梯度计算
     * 使用更优化的内存访问模式，便于编译器自动向量化
     */
    virtual void computeGradientsVectorized(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients) const;
    
    /** 
     * 带性能监控的损失计算
     * 用于性能分析和优化
     */
    virtual double computeBatchLossWithTiming(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        double& computeTimeMs) const;
    
    // =============================================
    // 性能工具方法
    // =============================================
    
    /** 
     * 估算计算复杂度（操作数）
     * 用于自动选择并行策略
     */
    virtual size_t estimateComputeOps(size_t sampleCount) const {
        return sampleCount; // 默认：每样本一次操作
    }
    
    /** 
     * 获取推荐的并行阈值
     * 小于此阈值时使用串行计算
     */
    virtual size_t getParallelThreshold() const {
        return 2000; // 默认阈值
    }
};

#endif // BOOSTING_LOSS_IREGRESSIONLOSS_HPP