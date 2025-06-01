// =============================================================================
// src/xgboost_main.cpp - 独立的XGBoost主程序
// =============================================================================
#include "xgboost/app/XGBoostApp.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void printBanner() {
    std::cout << "===============================================" << std::endl;
    std::cout << "           XGBoost Implementation             " << std::endl;
    std::cout << "     Extreme Gradient Boosting Trees          " << std::endl;
    std::cout << "===============================================" << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "\nUSAGE:" << std::endl;
    std::cout << "  " << programName << " [OPTIONS]" << std::endl;
    
    std::cout << "\nREQUIRED PARAMETERS:" << std::endl;
    std::cout << "  --data PATH           Training data CSV file path" << std::endl;
    
    std::cout << "\nMODEL PARAMETERS:" << std::endl;
    std::cout << "  --objective STR       Objective function (default: reg:squarederror)" << std::endl;
    std::cout << "                        Options: reg:squarederror, reg:logistic" << std::endl;
    std::cout << "  --num-rounds INT      Number of boosting rounds (default: 100)" << std::endl;
    std::cout << "  --eta FLOAT           Learning rate/shrinkage (default: 0.3)" << std::endl;
    std::cout << "  --max-depth INT       Maximum tree depth (default: 6)" << std::endl;
    std::cout << "  --min-child-weight INT Minimum sum of instance weight in child (default: 1)" << std::endl;
    
    std::cout << "\nREGULARIZATION PARAMETERS:" << std::endl;
    std::cout << "  --lambda FLOAT        L2 regularization parameter (default: 1.0)" << std::endl;
    std::cout << "  --gamma FLOAT         Minimum loss reduction for split (default: 0.0)" << std::endl;
    std::cout << "  --alpha FLOAT         L1 regularization parameter (default: 0.0)" << std::endl;
    
    std::cout << "\nSAMPLING PARAMETERS:" << std::endl;
    std::cout << "  --subsample FLOAT     Subsample ratio of training instances (default: 1.0)" << std::endl;
    std::cout << "  --colsample-bytree FLOAT Subsample ratio of columns by tree (default: 1.0)" << std::endl;
    
    std::cout << "\nTRAINING CONTROL:" << std::endl;
    std::cout << "  --early-stopping INT  Early stopping rounds (default: 0, disabled)" << std::endl;
    std::cout << "  --tolerance FLOAT     Convergence tolerance (default: 1e-7)" << std::endl;
    std::cout << "  --val-split FLOAT     Validation split ratio (default: 0.2)" << std::endl;
    std::cout << "  --verbose             Enable verbose output (default: true)" << std::endl;
    std::cout << "  --quiet               Disable verbose output" << std::endl;
    
    std::cout << "\nPERFORMANCE PARAMETERS:" << std::endl;
    std::cout << "  --approx-split        Use approximate split algorithm (default: false)" << std::endl;
    std::cout << "  --max-bins INT        Maximum number of bins for histograms (default: 256)" << std::endl;
    
    std::cout << "\nOTHER OPTIONS:" << std::endl;
    std::cout << "  --help, -h            Show this help message" << std::endl;
    std::cout << "  --version, -v         Show version information" << std::endl;
    
    std::cout << "\nEXAMPLES:" << std::endl;
    std::cout << "  Basic regression:" << std::endl;
    std::cout << "    " << programName << " --data ../data/data_clean/cleaned_data.csv" << std::endl;
    
    std::cout << "\n  High regularization (prevent overfitting):" << std::endl;
    std::cout << "    " << programName << " --data data.csv --num-rounds 200 --eta 0.1 \\" << std::endl;
    std::cout << "                       --max-depth 4 --lambda 5.0 --gamma 1.0" << std::endl;
    
    std::cout << "\n  Fast training with shallow trees:" << std::endl;
    std::cout << "    " << programName << " --data data.csv --num-rounds 500 --eta 0.5 \\" << std::endl;
    std::cout << "                       --max-depth 3 --lambda 0.1" << std::endl;
    
    std::cout << "\n  With early stopping:" << std::endl;
    std::cout << "    " << programName << " --data data.csv --early-stopping 20 \\" << std::endl;
    std::cout << "                       --val-split 0.2" << std::endl;
    
    std::cout << "\n  Logistic regression:" << std::endl;
    std::cout << "    " << programName << " --data data.csv --objective reg:logistic \\" << std::endl;
    std::cout << "                       --num-rounds 150 --eta 0.1" << std::endl;
}

void printVersion() {
    std::cout << "XGBoost Implementation v1.0.0" << std::endl;
    std::cout << "Built with C++17" << std::endl;
    std::cout << "Copyright (c) 2024" << std::endl;
}

bool parseArguments(int argc, char** argv, XGBoostAppOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            return false; // 触发help显示
        }
        else if (arg == "--version" || arg == "-v") {
            printVersion();
            exit(0);
        }
        else if (arg == "--data") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --data requires a value" << std::endl;
                return false;
            }
            opts.dataPath = argv[++i];
        }
        else if (arg == "--objective") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --objective requires a value" << std::endl;
                return false;
            }
            opts.objective = argv[++i];
        }
        else if (arg == "--num-rounds") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --num-rounds requires a value" << std::endl;
                return false;
            }
            try {
                opts.numRounds = std::stoi(argv[++i]);
                if (opts.numRounds <= 0) {
                    std::cerr << "Error: --num-rounds must be positive" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --num-rounds" << std::endl;
                return false;
            }
        }
        else if (arg == "--eta") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --eta requires a value" << std::endl;
                return false;
            }
            try {
                opts.eta = std::stod(argv[++i]);
                if (opts.eta <= 0.0 || opts.eta > 1.0) {
                    std::cerr << "Error: --eta must be in (0, 1]" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --eta" << std::endl;
                return false;
            }
        }
        else if (arg == "--max-depth") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --max-depth requires a value" << std::endl;
                return false;
            }
            try {
                opts.maxDepth = std::stoi(argv[++i]);
                if (opts.maxDepth <= 0) {
                    std::cerr << "Error: --max-depth must be positive" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --max-depth" << std::endl;
                return false;
            }
        }
        else if (arg == "--min-child-weight") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --min-child-weight requires a value" << std::endl;
                return false;
            }
            try {
                opts.minChildWeight = std::stoi(argv[++i]);
                if (opts.minChildWeight < 0) {
                    std::cerr << "Error: --min-child-weight must be non-negative" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --min-child-weight" << std::endl;
                return false;
            }
        }
        else if (arg == "--lambda") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --lambda requires a value" << std::endl;
                return false;
            }
            try {
                opts.lambda = std::stod(argv[++i]);
                if (opts.lambda < 0.0) {
                    std::cerr << "Error: --lambda must be non-negative" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --lambda" << std::endl;
                return false;
            }
        }
        else if (arg == "--gamma") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --gamma requires a value" << std::endl;
                return false;
            }
            try {
                opts.gamma = std::stod(argv[++i]);
                if (opts.gamma < 0.0) {
                    std::cerr << "Error: --gamma must be non-negative" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --gamma" << std::endl;
                return false;
            }
        }
        else if (arg == "--subsample") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --subsample requires a value" << std::endl;
                return false;
            }
            try {
                opts.subsample = std::stod(argv[++i]);
                if (opts.subsample <= 0.0 || opts.subsample > 1.0) {
                    std::cerr << "Error: --subsample must be in (0, 1]" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --subsample" << std::endl;
                return false;
            }
        }
        else if (arg == "--colsample-bytree") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --colsample-bytree requires a value" << std::endl;
                return false;
            }
            try {
                opts.colsampleByTree = std::stod(argv[++i]);
                if (opts.colsampleByTree <= 0.0 || opts.colsampleByTree > 1.0) {
                    std::cerr << "Error: --colsample-bytree must be in (0, 1]" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --colsample-bytree" << std::endl;
                return false;
            }
        }
        else if (arg == "--early-stopping") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --early-stopping requires a value" << std::endl;
                return false;
            }
            try {
                opts.earlyStoppingRounds = std::stoi(argv[++i]);
                if (opts.earlyStoppingRounds < 0) {
                    std::cerr << "Error: --early-stopping must be non-negative" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --early-stopping" << std::endl;
                return false;
            }
        }
        else if (arg == "--tolerance") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --tolerance requires a value" << std::endl;
                return false;
            }
            try {
                opts.tolerance = std::stod(argv[++i]);
                if (opts.tolerance < 0.0) {
                    std::cerr << "Error: --tolerance must be non-negative" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --tolerance" << std::endl;
                return false;
            }
        }
        else if (arg == "--val-split") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --val-split requires a value" << std::endl;
                return false;
            }
            try {
                opts.valSplit = std::stod(argv[++i]);
                if (opts.valSplit < 0.0 || opts.valSplit >= 1.0) {
                    std::cerr << "Error: --val-split must be in [0, 1)" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --val-split" << std::endl;
                return false;
            }
        }
        else if (arg == "--max-bins") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --max-bins requires a value" << std::endl;
                return false;
            }
            try {
                opts.maxBins = std::stoi(argv[++i]);
                if (opts.maxBins <= 0) {
                    std::cerr << "Error: --max-bins must be positive" << std::endl;
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid value for --max-bins" << std::endl;
                return false;
            }
        }
        else if (arg == "--verbose") {
            opts.verbose = true;
        }
        else if (arg == "--quiet") {
            opts.verbose = false;
        }
        else if (arg == "--approx-split") {
            opts.useApproxSplit = true;
        }
        else {
            std::cerr << "Error: Unknown argument '" << arg << "'" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool validateOptions(const XGBoostAppOptions& opts) {
    if (opts.dataPath.empty()) {
        std::cerr << "Error: --data is required" << std::endl;
        return false;
    }
    
    // 验证目标函数
    if (opts.objective != "reg:squarederror" && 
        opts.objective != "reg:logistic" &&
        opts.objective != "reg:linear" &&
        opts.objective != "binary:logistic") {
        std::cerr << "Error: Unsupported objective '" << opts.objective << "'" << std::endl;
        std::cerr << "Supported objectives: reg:squarederror, reg:logistic, reg:linear, binary:logistic" << std::endl;
        return false;
    }
    
    return true;
}

void printConfiguration(const XGBoostAppOptions& opts) {
    std::cout << "\nXGBoost Configuration:" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << std::left;
    std::cout << std::setw(25) << "Data Path:" << opts.dataPath << std::endl;
    std::cout << std::setw(25) << "Objective:" << opts.objective << std::endl;
    std::cout << std::setw(25) << "Number of Rounds:" << opts.numRounds << std::endl;
    std::cout << std::setw(25) << "Learning Rate (eta):" << opts.eta << std::endl;
    std::cout << std::setw(25) << "Max Depth:" << opts.maxDepth << std::endl;
    std::cout << std::setw(25) << "Min Child Weight:" << opts.minChildWeight << std::endl;
    std::cout << std::setw(25) << "Lambda (L2):" << opts.lambda << std::endl;
    std::cout << std::setw(25) << "Gamma (min split loss):" << opts.gamma << std::endl;
    std::cout << std::setw(25) << "Subsample:" << opts.subsample << std::endl;
    std::cout << std::setw(25) << "Column Sample by Tree:" << opts.colsampleByTree << std::endl;
    
    if (opts.earlyStoppingRounds > 0) {
        std::cout << std::setw(25) << "Early Stopping Rounds:" << opts.earlyStoppingRounds << std::endl;
        std::cout << std::setw(25) << "Validation Split:" << opts.valSplit << std::endl;
    }
    
    if (opts.useApproxSplit) {
        std::cout << std::setw(25) << "Split Algorithm:" << "Approximate" << std::endl;
        std::cout << std::setw(25) << "Max Bins:" << opts.maxBins << std::endl;
    } else {
        std::cout << std::setw(25) << "Split Algorithm:" << "Exact" << std::endl;
    }
    
    std::cout << std::setw(25) << "Verbose:" << (opts.verbose ? "Yes" : "No") << std::endl;
    std::cout << std::setw(25) << "Tolerance:" << opts.tolerance << std::endl;
    std::cout << std::endl;
}

void printParameterTips(const XGBoostAppOptions& opts) {
    std::cout << "\nParameter Tuning Tips:" << std::endl;
    std::cout << "=====================" << std::endl;
    
    if (opts.eta > 0.5) {
        std::cout << "- Consider reducing learning rate (eta) to < 0.3 for better convergence" << std::endl;
    }
    
    if (opts.maxDepth > 10) {
        std::cout << "- Deep trees (depth > 10) may cause overfitting, consider max-depth 6-8" << std::endl;
    }
    
    if (opts.lambda < 0.1) {
        std::cout << "- Low regularization may cause overfitting, consider lambda >= 1.0" << std::endl;
    }
    
    if (opts.numRounds > 500) {
        std::cout << "- Many rounds may cause overfitting, consider early stopping" << std::endl;
    }
    
    if (opts.earlyStoppingRounds == 0 && opts.numRounds > 100) {
        std::cout << "- Consider enabling early stopping with --early-stopping 20" << std::endl;
    }
    
    std::cout << "\nRegularization Guidelines:" << std::endl;
    std::cout << "- For small datasets: increase lambda (2.0-10.0), reduce max-depth (3-4)" << std::endl;
    std::cout << "- For large datasets: moderate lambda (0.1-2.0), moderate max-depth (6-8)" << std::endl;
    std::cout << "- To prevent overfitting: lower eta (0.05-0.1), enable early stopping" << std::endl;
    std::cout << "- For fast prototyping: higher eta (0.3-0.5), shallow trees (3-4)" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    printBanner();
    
    if (argc == 1) {
        std::cout << "No arguments provided. Use --help for usage information." << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // 设置默认选项
    XGBoostAppOptions opts;
    opts.dataPath = "";  // 必须由用户指定
    opts.objective = "reg:squarederror";
    opts.numRounds = 100;
    opts.eta = 0.3;
    opts.maxDepth = 6;
    opts.minChildWeight = 1;
    opts.lambda = 1.0;
    opts.gamma = 0.0;
    opts.subsample = 1.0;
    opts.colsampleByTree = 1.0;
    opts.verbose = true;
    opts.earlyStoppingRounds = 0;
    opts.tolerance = 1e-7;
    opts.valSplit = 0.2;
    opts.useApproxSplit = false;
    opts.maxBins = 256;
    
    // 解析命令行参数
    if (!parseArguments(argc, argv, opts)) {
        printUsage(argv[0]);
        return 1;
    }
    
    // 验证参数
    if (!validateOptions(opts)) {
        return 1;
    }
    
    try {
        // 显示配置
        if (opts.verbose) {
            printConfiguration(opts);
            printParameterTips(opts);
        }
        
        // 运行XGBoost训练
        runXGBoostApp(opts);
        
        if (opts.verbose) {
            std::cout << "\nTraining completed successfully!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nError during training: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nUnknown error occurred during training" << std::endl;
        return 1;
    }
    
    return 0;
}