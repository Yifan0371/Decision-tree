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
