# pylidar 开发任务

## 背景

将 R 语言 `lidR` 包（/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR）的单木分割（Individual Tree Segmentation, ITS）算法迁移到 Python 生态，最终发布一个可通过 `pip install pylidar` 安装的包。

**核心策略：**
- 算法核心层用纯 C++17 实现，去除所有 R/Rcpp 依赖
- 通过 pybind11 或者 nanobind 暴露给 Python
- 构建后端用 scikit-build-core
- 通过 cibuildwheel 在 GitHub Actions 上构建多平台 wheel（Linux x86_64/aarch64、macOS x86_64/arm64、Windows）

**当前阶段只迁移 ITS 相关算法**。后续会逐步迁移 lidR 中其他算法（地面分割、DTM 插值、点云度量、波形处理等），所以**架构必须为后续扩展预留扩展点**——core 层应该按算法族（its/itd/dtm/metrics/...）分目录，而不是把所有算法平铺。

## 范围

本期需要迁移的算法（lidR 源码：/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR）：

| 算法 | 源文件 | 输入 | 优先级 |
|---|---|---|---|
| `dalponte2016` | `C_dalponte2016.cpp` | CHM + seeds（两个 2D raster） | P0 |
| `silva2016` | `C_silva2016.cpp` | CHM + seeds | P0 |
| `li2012` | `C_li2012.cpp` + `SpatialIndex.cpp` + `Point.h` | 点云 (Nx3) | P1 |
| `lmf`（树顶检测） | `C_lmf.cpp` | CHM 或点云 | 待 brainstorm 决定 |
| `chm_smooth` | `C_smooth.cpp` | CHM | 可选 |
| `watershed` | （R 层调 EBImage） | CHM | 待 brainstorm 决定 |

## 关键架构约束

1. **License 是 GPL-3**。lidR 本身是 GPL-3，移植/fork 它的代码 pylidar 也必须 GPL-3，`pyproject.toml` 中要明确标注。

2. **三层分离的目录结构**（最终形态，本期只填 ITS 部分，其他目录留空或不建）：
```
   pylidar/
   ├── pyproject.toml
   ├── CMakeLists.txt
   ├── src/
   │   ├── core/                 # 纯 C++，零 Python 依赖，可独立链接到任意 C++ 项目
   │   │   ├── common/
   │   │   │   ├── point.hpp
   │   │   │   ├── matrix2d.hpp        # 替代 Rcpp::NumericMatrix（列主序）
   │   │   │   └── spatial_index.hpp/cpp
   │   │   └── its/
   │   │       ├── dalponte2016.hpp/cpp
   │   │       ├── silva2016.hpp/cpp
   │   │       └── li2012.hpp/cpp
   │   └── bindings/             # pybind11 胶水层
   │       └── module.cpp
   ├── python/pylidar/
   │   ├── __init__.py
   │   ├── _core.pyi             # 类型存根
   │   └── segmentation.py       # 高级 Python API
   ├── tests/
   └── .github/workflows/wheels.yml
```

3. **去 Rcpp 化映射表**：
   - `Rcpp::NumericMatrix` / `IntegerMatrix` → 自实现的 `Matrix2D<T>`（**列主序**，与 lidR/R 一致，最大化代码复用），**不引入 Eigen 等重依赖**
   - `Rcpp::stop(...)` → `throw std::invalid_argument(...)` 或 `std::runtime_error(...)`
   - `Rcpp::Rcout` / `Rcerr` → 删除，或换成可选的日志回调 `std::function<void(const std::string&)>`
   - `Rcpp::checkUserInterrupt()` → 删除（Python 层由用户用 `signal` 处理 Ctrl-C）
   - `R_NaN` / `NA_REAL` → `std::numeric_limits<double>::quiet_NaN()`
   - `S4 las.slot("data")` → 改为接收 `const std::vector<double>&` 或 `const double*` + 长度
   - `Rcpp::List` 返回 → 自定义 POD struct
   - OpenMP `#pragma omp` → 是否保留待 brainstorm 决定

4. **数据接口设计**：
   - core 层只用 `Matrix2D<T>`、`std::vector<T>`、`std::span<T>`（C++20 不上则用裸指针 + size）这类零依赖类型
   - bindings 层用 `py::array_t<...>` 与 numpy 零拷贝交互
   - Python 用户面向的是 **row-major** numpy 数组，列主序转换在 bindings 层完成
   - **不复刻 R 的 LAS S4 对象**。LAS/LAZ 文件读取交给 `laspy`（pip 直装，BSD-2），pylidar 只接受 numpy 数组输入

5. **测试策略**：每个算法至少要有
   - 端到端测试（最小合成数据，数值结果与 lidR 输出在容差内一致）
   - 边界条件测试（空输入、单点、退化输入）
   - 性能基准（与 lidR 跑相同输入的耗时对比，作为后续优化依据）

## 沟通约定

- **重大设计决策**（影响多文件、改变 API、引入新依赖）必须先问我
- **小决策**（变量命名、注释风格、单文件内部重构）自己定，不要打断
- 遇到 lidR 源码里看不懂的部分，**直接读 GitHub 上的 cpp 文件**而不是猜

## 起点

- lidR 仓库：https://github.com/r-lidar/lidR
- lidR 本地仓库地址：/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR
- 重点目录：https://github.com/r-lidar/lidR/tree/master/src
- 本地工作目录：当前 `pylidar/` 文件夹
