# Decision-tree


## 依赖

* CMake ≥ 3.10
* 支持 C++17 的编译器（GCC/Clang/MSVC）

## 克隆仓库

```bash
git clone 
cd Decision-tree
```

## 构建项目

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

此时可执行文件将位于 `build/bin/DataCleanApp`。

## 运行数据清洗

```bash
# 若当前在 build/ 目录：
./bin/DataCleanApp

# 若在项目根目录：
build/bin/DataCleanApp
```

清洗结果会输出到 `data/data_clean/cleaned_<原文件名>.csv`。

# 单棵决策树测试脚本
# 1. 先编译项目（在项目根目录）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# 2. 复制可执行文件到根目录
cp build/DecisionTreeMain .

# 3. 进入脚本目录
cd script/single_tree

# 4. 给脚本添加执行权限
chmod +x *.sh

# 5. 运行脚本
./test_criterion.sh
./test_finder.sh
./test_pruner.sh

# 重新运行测试
bash script/boosting/gbrt/test_gbrt_comprehensive.sh