#include "pipeline/DataSplit.hpp"

bool splitDataset(const std::vector<double>& X,
                  const std::vector<double>& y,
                  int rowLength,
                  DataParams& out) {
    // DataIO 给的 rowLength = features+1
    int feat = rowLength - 1;
    size_t totalRows = y.size();
    size_t trainRows = static_cast<size_t>(totalRows * 0.8);

    out.rowLength = feat;
    out.X_train.assign(X.begin(), X.begin() + trainRows * feat);
    out.y_train.assign(y.begin(), y.begin() + trainRows);
    out.X_test.assign(X.begin() + trainRows * feat, X.end());
    out.y_test.assign(y.begin() + trainRows, y.end());
    return true;
}
