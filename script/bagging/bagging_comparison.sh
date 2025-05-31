#!/bin/bash

# Bagging Performance Comparison Script
# Comparing Exhaustive vs Random Split Finders
# Configuration: MSE criterion, No pruning

echo "=========================================="
echo "Bagging Performance Comparison"
echo "=========================================="
echo "Comparing: Exhaustive vs Random Finders"
echo "Configuration: MSE criterion, No pruner"
echo "Date: $(date)"
echo ""

# Create results directory
RESULTS_DIR="script/bagging"
mkdir -p $RESULTS_DIR

# Result files
EXHAUSTIVE_RESULTS="$RESULTS_DIR/exhaustive_results.txt"
RANDOM_RESULTS="$RESULTS_DIR/random_results.txt"

# Clear previous results
> $EXHAUSTIVE_RESULTS
> $RANDOM_RESULTS

# Write file headers
echo "# Exhaustive Finder Results - $(date)" > $EXHAUSTIVE_RESULTS
echo "# Format: Trees, Train_Time(ms), Test_MSE, Test_MAE, OOB_MSE" >> $EXHAUSTIVE_RESULTS
echo "Trees,Train_Time,Test_MSE,Test_MAE,OOB_MSE" >> $EXHAUSTIVE_RESULTS

echo "# Random Finder Results - $(date)" > $RANDOM_RESULTS
echo "# Format: Trees, Train_Time(ms), Test_MSE, Test_MAE, OOB_MSE" >> $RANDOM_RESULTS
echo "Trees,Train_Time,Test_MSE,Test_MAE,OOB_MSE" >> $RANDOM_RESULTS

# Test configurations
TREE_COUNTS=(5 10 20 50 100)
DATA_PATH="data/data_clean/cleaned_data.csv"
EXECUTABLE="build/DecisionTreeMain"

# Check executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: $EXECUTABLE not found!"
    echo "Please run 'make' in the build directory first."
    exit 1
fi

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file $DATA_PATH not found!"
    exit 1
fi

echo "Testing configurations:"
echo "Tree counts: ${TREE_COUNTS[*]}"
echo "Data: $DATA_PATH"
echo "Timeout: 300 seconds per test"
echo ""

# Function to extract results
extract_results() {
    local log_file=$1
    local train_time=$(grep "Train Time:" $log_file | awk '{print $3}' | sed 's/ms//')
    local test_mse=$(grep "Test MSE:" $log_file | awk '{print $3}')
    local test_mae=$(grep "Test MAE:" $log_file | awk '{print $6}')
    local oob_mse=$(grep "OOB MSE:" $log_file | awk '{print $3}')
    
    echo "$train_time,$test_mse,$test_mae,$oob_mse"
}

# Main test loop
for trees in "${TREE_COUNTS[@]}"; do
    echo "Testing with $trees trees..."
    
    # Test Exhaustive Finder
    echo "  Running Exhaustive finder..."
    temp_log="temp_exhaustive_${trees}.log"
    
    timeout 300 $EXECUTABLE bagging $DATA_PATH $trees 1.0 30 2 mse exhaustive none 0.01 42 > $temp_log 2>&1
    
    if [ $? -eq 0 ]; then
        results=$(extract_results $temp_log)
        echo "$trees,$results" >> $EXHAUSTIVE_RESULTS
        echo "    Exhaustive completed: $results"
    else
        echo "    Exhaustive failed or timeout"
        echo "$trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> $EXHAUSTIVE_RESULTS
    fi
    
    # Test Random Finder
    echo "  Running Random finder..."
    temp_log="temp_random_${trees}.log"
    
    timeout 300 $EXECUTABLE bagging $DATA_PATH $trees 1.0 30 2 mse random none 0.01 42 > $temp_log 2>&1
    
    if [ $? -eq 0 ]; then
        results=$(extract_results $temp_log)
        echo "$trees,$results" >> $RANDOM_RESULTS
        echo "    Random completed: $results"
    else
        echo "    Random failed or timeout"
        echo "$trees,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> $RANDOM_RESULTS
    fi
    
    # Clean temporary files
    rm -f temp_exhaustive_${trees}.log temp_random_${trees}.log
    
    echo "  Completed $trees trees test"
    echo ""
done

echo "=========================================="
echo "Summary Report"
echo "=========================================="

echo ""
echo "Exhaustive Finder Results:"
echo "Trees | Train_Time | Test_MSE | Test_MAE | OOB_MSE"
echo "------|------------|----------|----------|--------"
tail -n +4 $EXHAUSTIVE_RESULTS | while IFS=',' read trees train_time test_mse test_mae oob_mse; do
    printf "%5s | %10s | %8s | %8s | %7s\n" "$trees" "$train_time" "$test_mse" "$test_mae" "$oob_mse"
done

echo ""
echo "Random Finder Results:"
echo "Trees | Train_Time | Test_MSE | Test_MAE | OOB_MSE"
echo "------|------------|----------|----------|--------"
tail -n +4 $RANDOM_RESULTS | while IFS=',' read trees train_time test_mse test_mae oob_mse; do
    printf "%5s | %10s | %8s | %8s | %7s\n" "$trees" "$train_time" "$test_mse" "$test_mae" "$oob_mse"
done

echo ""
echo "Results saved to:"
echo "  - $EXHAUSTIVE_RESULTS"
echo "  - $RANDOM_RESULTS"
echo ""
echo "Performance comparison completed."
echo "=========================================="