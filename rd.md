# pylidar 开发任务

## 背景

将 R 语言 `lidR` 包（本地：`/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR`，上游：https://github.com/r-lidar/lidR）的单木分割（Individual Tree Segmentation, ITS）算法迁移到 Python 生态。

**当前阶段目标：本地 Python 可调用即可，不发布 PyPI、不做多平台 wheel 分发。** 通过 `pip install -e .` 在本机以可编辑模式安装，能够 `import pylidar` 调用算法即视为完成。

**核心策略：**
- 算法核心层用纯 C++17 实现，去除所有 R/Rcpp 依赖
- 通过 **nanobind** 暴露给 Python（相比 pybind11，二进制更小、编译更快、运行时开销更低）
- **产物只有一个 Python 扩展模块** `pylidar/_core.<abi>.so`（editable install 直接放在源码树下），**只能通过 `import pylidar._core` 使用，不是可被其他 C++ 程序 link 的独立 dylib**——它只对 CPython 暴露 `PyInit__core` 入口，内部 C++ 函数不在导出符号表里。core 层 C++ 源码保持零 Python 依赖，是为了单元测试隔离 + 未来可移植，**不代表本期会单独输出 C++ 库**
- 构建后端用 **scikit-build-core**，与 nanobind 配合，在 macOS 和 Linux 上都能本地 `pip install -e .` 编辑模式安装
- **目标平台：macOS（Apple Silicon）+ Linux（Ubuntu）**，两个平台都要能本地 `pip install -e .` 跑通并通过测试；Windows 本期不支持。C++ 代码不调平台特有 API（POSIX 之外的能不用就不用），CMake 不硬编码平台路径或编译器

**当前阶段只迁移 ITS 相关算法**，但本期是 lidR 算法迁移的起点——后续会逐步迁入 lidR 中其他算法族（地面分割、DTM 插值、点云度量、波形处理等）。所以**目录结构和 CMake 构建从第一行代码就要为多算法族扩展打开口子**：

- core 层按算法族分目录（`its/`、`itd/`、`dtm/`、`metrics/`...），不要把所有算法平铺到一个 `core/` 下
- CMake 里每个算法族独立成一个静态库子目标（如 `pylidar_core_its`），`bindings/module.cpp` 按需链接；新增算法族就是加一个子目录 + 一行 `add_subdirectory`，不需要改动既有目标
- Python 层 `python/pylidar/` 也按算法族分子模块（如 `pylidar.segmentation`、未来的 `pylidar.ground`、`pylidar.metrics`），避免所有 API 挤在一个大 namespace 里

## 范围

**重要更正**：核对 lidR 源码后发现，原计划里若干"以为是独立 .cpp 文件"的算法实际上要么是纯 R 实现、要么是 `LAS` god-class 的方法。重新分桶如下：

### 桶一：要写 C++ 的算法

| 算法 | lidR 源位置 | 输入 | 优先级 | 备注 |
|---|---|---|---|---|
| `dalponte2016` | `src/C_dalponte2016.cpp`（独立文件） | CHM + seeds（两个 2D raster） | **P0** | 唯一一个干净独立的 C++ 文件，flood-fill，无空间索引依赖 |
| `li2012` | `src/RcppFunction.cpp:92` 入口 + `src/LAS.cpp:1113` 的 `LAS::segment_trees`（~150 行） | 点云 (Nx3) | P1 | 与 LAS 类纠缠，需先抽出独立的 `PointXYZ` 容器 |
| `lmf`（树顶检测） | `src/RcppFunction.cpp:37` 入口 + `src/LAS.cpp` 的 `LAS::filter_local_maxima`（两个重载，line 399 / 482） | 点云 或 CHM | P1 | 需要 2D 半径搜索 → 见空间索引策略 |
| `chm_smooth` | `src/RcppFunction.cpp:45` 入口 + `src/LAS.cpp:112` 的 `LAS::z_smooth` | 点云 | P2 | 需要邻域查询，同上 |

### 桶二：纯 Python 实现就够（无 C++ 工作量）

| 算法 | lidR 源位置 | Python 实现思路 | 优先级 |
|---|---|---|---|
| `silva2016` | `R/algorithm-its.R:203`（**纯 R，无 C++**） | 本质就是「KNN + 两个阈值过滤」，`scipy.spatial.cKDTree` + 几行 NumPy ≈ 30 行 | **P0** |
| `watershed` | `R/algorithm-its.R`（调 Bioconductor `EBImage::watershed`） | `skimage.segmentation.watershed`，~5 行包装 | P2 |

### 桶三：基础设施

- `Matrix2D<T>`：自实现的 2D 矩阵容器（**row-major**，详见架构约束 §4）
- `PointXYZ`：轻量 POD struct，从 `inst/include/lidR/Point.h` 简化迁移
- 空间索引：**直接 vendor nanoflann**（`src/nanoflann/nanoflann.h`，header-only，BSD-2，与 GPL-3 兼容）。**不**自写 QuadTree / GridPartition——lidR 的实现深度耦合 Rcpp（`Rcpp::S4 las` / `NumericVector` 在构造函数和存储里），从那里反向解耦的成本远高于直接用 nanoflann 的 KD-tree

### 不在本期的算法（提前记一笔，避免架构被未来需求打破）

未来要迁的 `point_metrics` / 特征值分解相关算法依赖 lidR 的 `LinkingTo: BH (>= 1.72.0), Rcpp, RcppArmadillo`。本期 ITS 用不到 Boost/Armadillo，但**目录结构与 CMake 要给"将来引入 Eigen 替代 Armadillo"留好位置**（在 `src/core/common/` 下按需加 `linalg/`，不引入到本期编译目标里）。

## 关键架构约束

1. **License 是 GPL-3**。lidR 本身是 GPL-3，移植/fork 它的代码 pylidar 也必须 GPL-3，`pyproject.toml` 与仓库根目录 LICENSE 文件中明确标注。
   
   - **下游传染性提示**：任何 `import pylidar` 的发行物受 GPL-3 约束。如果将来想做"宽松 license 的核心 + GPL 包装层"，核心代码必须**独立写就**、不能从 lidR 派生——本期不打算做，记一笔避免未来糊涂。
   
2. **三层分离的目录结构**（最终形态，本期只填 ITS 部分，其他目录留空或不建）：
   ```
   pylidar/
   ├── pyproject.toml
   ├── CMakeLists.txt
   ├── LICENSE                       # GPL-3 全文
   ├── src/
   │   ├── core/                     # 纯 C++ 源码，零 Python 依赖（编为静态库，链入 _core 扩展；不单独产出 .so/.dylib）
   │   │   ├── common/
   │   │   │   ├── point.hpp         # PointXYZ 等 POD
   │   │   │   ├── matrix2d.hpp      # 行主序，与 numpy 一致
   │   │   │   └── kdtree.hpp        # 对 vendored nanoflann 的薄包装
   │   │   └── its/
   │   │       ├── dalponte2016.hpp/cpp
   │   │       └── li2012.hpp/cpp     # M3 才填
   │   ├── third_party/
   │   │   └── nanoflann/
   │   │       └── nanoflann.h        # vendored，保留原 LICENSE 文件
   │   └── bindings/                  # nanobind 胶水层
   │       └── module.cpp
   ├── python/pylidar/
   │   ├── __init__.py
   │   ├── _core.pyi                  # 类型存根（用 nanobind.stubgen 生成）
   │   └── segmentation.py            # 高级 Python API；silva2016 等纯 Python 算法也住在这里
   ├── tests/
   │   └── fixtures/                  # lidR 跑出的参考输出（.npz）
   └── tools/
       └── regen_fixtures.R           # 在 R 里重生成 fixtures 的脚本（不进 CI，留档）
   ```

3. **去 Rcpp 化映射表**：
   - `Rcpp::NumericMatrix` / `IntegerMatrix` → 自实现的 `Matrix2D<T>`（**row-major**，与 numpy 一致），**不引入 Eigen 等重依赖**
   - `Rcpp::stop(...)` → `throw std::invalid_argument(...)` 或 `std::runtime_error(...)`
   - `Rcpp::Rcout` / `Rcerr` → 删除，或换成可选的日志回调 `std::function<void(std::string_view)>`
   - `Rcpp::checkUserInterrupt()` → 删除（Python 层由用户用 `signal` 处理 Ctrl-C）
   - `R_NaN` / `NA_REAL` → `std::numeric_limits<double>::quiet_NaN()`
   - `S4 las.slot("data")` → 改为接收 `const double*` + `std::size_t` 长度，或 `const std::vector<double>&`（C++17 无 `std::span`）
   - `Rcpp::List` 返回 → 自定义 POD struct
   - `#pragma omp` → **本期一律删除/不开**，详见 §6

4. **数据接口设计**：
   - core 层只用 `Matrix2D<T>`、`std::vector<T>`、裸指针 + size 这类零依赖类型（C++17 无 `std::span`，要"非拥有视图"语义就明确用 `const T* ptr, std::size_t n` 双参数）
   - bindings 层用 nanobind 的 `nb::ndarray<...>` 与 numpy **零拷贝**交互
   - **行主序统一**：Python 用户面向的是 row-major numpy 数组，core 层 `Matrix2D<T>` 也是 row-major，bindings 层不做布局转换。lidR 是列主序但算法（dalponte2016 是 flood-fill / li2012 是点距离）对方向无偏好，逐元素翻译循环即可
   - bindings 接口**严格约束 dtype + 连续性**，避免隐式拷贝：
     - CHM：`nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>`
     - Seeds / 输出 region：`nb::ndarray<int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>`
     - 点云 X/Y/Z：`nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>`
   - **不复刻 R 的 LAS S4 对象**。LAS/LAZ 文件读取交给 `laspy`（pip 直装，BSD-2），pylidar 只接受 numpy 数组输入

5. **测试与 fixture 策略**：
   - 每个算法至少要有：端到端测试（最小合成数据，结果与 lidR 输出在容差内一致）+ 边界条件测试（空输入、单点、退化输入）+ 性能基准（与 lidR 跑相同输入耗时对比）
   - **fixture 流程必须固化**：
     - `tests/fixtures/*.npz` 提交 lidR 跑出的参考结果（小型合成 CHM + seeds，~50KB 量级）
     - `tools/regen_fixtures.R` 留一份 R 脚本，记录"如何重生成"——三个月后没人记得 reference 怎么算出来的，回归测试就废了
     - CI（如果以后开）只加载 `.npz` 做断言，不依赖 R 环境

6. **OpenMP 策略**：
   - **本期不开**。dalponte2016 跑 1000×1000 CHM 在毫秒级，单线程足够；其它算法首版也优先正确性。
   - macOS 上的实际坑：Apple Clang 默认不带 libomp，需要 `brew install libomp` + CMake 走 Homebrew 路径。把这条挡在本期之外能省调试时间。
   - CMake 里写 `find_package(OpenMP COMPONENTS CXX)` 但**不强求**，target 用 `if(OpenMP_CXX_FOUND) target_link_libraries(... OpenMP::OpenMP_CXX) target_compile_definitions(... PYLIDAR_HAS_OPENMP=1) endif()`。算法源码用 `#ifdef PYLIDAR_HAS_OPENMP` 包住未来要加的 `#pragma`。本期所有 `#pragma omp` **直接删除**，不留 `#ifdef` 死代码。

7. **算法行为对齐策略**：
   - 默认**先 1:1 行为对齐 lidR**（包括它的小怪癖、O(N²) 距离扫描这种已知非最优实现），通过 fixture 测试后再考虑性能/可读性重构
   - 例：`li2012` 内部对每个 u 都做一次全量 sqdistance（`LAS.cpp:1207`），明显有空间索引可优化空间，但首版**别动**；先对齐再说
   - 这条是为了让 fixture diff 干净；优化和"行为变化"必须是两个独立 commit

## 构建与工具链

### Python 与依赖最低版本

`pyproject.toml` 显式 pin（确认在 Python 3.14 上有 wheel/支持）：

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.4"]
build-backend = "scikit_build_core.build"

[project]
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.1",
    "laspy>=2.5",
    "scipy>=1.13",        # silva2016 / 通用 KNN
    "scikit-image>=0.24", # watershed
]
```

### scikit-build-core editable 模式必须开自动重编

默认 editable 安装**不会**随 .cpp 修改自动重编（改了 cpp 后 import 还跑老代码，是隐蔽坑）。`pyproject.toml` 里：

```toml
[tool.scikit-build]
build-dir = "build/{wheel_tag}"
editable.rebuild = true
editable.mode = "redirect"
cmake.version = ">=3.26"
cmake.build-type = "Release"
```

本地装：`pip install --no-build-isolation -e .`（**不**用 `uv pip install -e .`，详见下条）。

### uv 的使用边界

- `uv` 用于：管理 Python 解释器版本（`.python-version`）、虚拟环境（`.venv`）、纯 Python 依赖锁定
- `uv` **不**用于：editable install C++ 扩展（uv 对 scikit-build-core 的 editable rebuild 钩子支持目前比 pip 弱，会出现 cpp 改了但模块没重编的迷惑现象）
- 因此本地开发流程是混合的：`uv sync` → `pip install --no-build-isolation -e .`（在 uv 创建的 .venv 里）

### 跨平台前提

- macOS：Xcode Command Line Tools（提供 Apple Clang，C++17 全特性支持）
- Ubuntu：`build-essential` + `cmake>=3.26` + GCC 9+ 或 Clang 10+（C++17 完整支持的最低版本，Ubuntu 20.04 之后默认 GCC 9+ 都满足）

## 沟通约定

- **重大设计决策**（影响多文件、改变 API、引入新依赖）必须先问我
- **小决策**（变量命名、注释风格、单文件内部重构）自己定，不要打断
- 遇到 lidR 源码里看不懂的部分，**直接读 GitHub 上的 cpp 文件**而不是猜
- 对 lidR 的 bug fix / perf fix 优先**回灌上游**（作者 Jean-Romain Roussel 仍在维护），保持协作姿态

## 起点

- lidR 仓库：https://github.com/r-lidar/lidR
- lidR 本地仓库地址：/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR
- 重点目录：https://github.com/r-lidar/lidR/tree/master/src
- 本地工作目录：当前 `pylidar/` 文件夹

## 验收

在 macOS 和 Ubuntu 任一平台（**两个平台都跑通即发版**）上：

1. `pip install -e .` 成功，无平台特异编译错误
2. `python -c "import pylidar"` 不报错
3. P0 算法（**dalponte2016 + silva2016**）端到端测试通过，输出与 lidR fixture 在 `np.allclose(rtol=1e-6)` 容差内一致

## 项目命名

`pylidar` 这个名字在 PyPI 上已被一个 SPDLib-based LiDAR 处理库占用。本期不发 PyPI 不影响。**如果将来要发**，需提前换名，候选：`pylidR`、`lidR-py`、`treesegpy`、`rs-lidar`。在做发布决定时再定，不本期阻塞。
