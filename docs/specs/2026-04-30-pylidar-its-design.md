# pylidar — 个体树分割（ITS）算法迁移设计

- 状态：草案，已通过 brainstorming 阶段评审
- 日期：2026-04-30
- 范围版本：v0.1（仅 ITS 算法族）
- License：GPL-3.0-or-later

## 1. 摘要

本设计将 R 语言 `lidR` 包（v4.3.2）中的个体树分割算法迁移到一个独立的 Python 可 pip 安装的库 `pylidar`：纯 C++17 算法核心 + nanobind Python 绑定 + scikit-build-core/cibuildwheel 多平台 wheel。当前阶段仅迁移 ITS 算法族（dalponte2016、silva2016、li2012、lmf 树顶检测、smooth_height 点云高度平滑、watershed），但目录与构建结构按"未来扩展到 DTM / metrics / waveform 等多个算法族"的需要预留扩展点。LAS/LAZ 文件 IO 不在范围内。

## 2. 背景与目标

`lidR` 是航空激光雷达林业处理事实上的参考实现，但 R/Rcpp 耦合使其难以直接用于 Python 的 ML/3D 流水线。需求方需要在 Python 端无 R 依赖地获得相同算法。`pylidar` 的目标：

- **正确性优先**：算法行为应可与 lidR 在容差内对齐。
- **独立可链接**：C++ core 可被任意 C++ 项目链接（不依赖 R、不依赖 Python）。
- **打包友好**：用户 `pip install pylidar` 即可，不需要本机有 R/编译器。
- **架构可扩展**：本期只填 ITS，后期增加 DTM、ground、metrics、waveform 等算法族应是"加文件而非改架构"。

## 3. 范围

### 3.1 In-scope（v0.1）

| 算法 | Python 公共名 | 输入 | 输出 |
|---|---|---|---|
| Dalponte 2016 | `segment_dalponte2016` | CHM 栅格 (float64) + 树顶 | 树冠 ID 栅格 (H,W) int32 |
| Silva 2016 | `segment_silva2016` | CHM 栅格 (float64) + 树顶 | 树冠 ID 栅格 (H,W) int32 |
| Li 2012 | `segment_li2012` | XYZ 点云 (N,3) float64 | 每点 ID (N,) int32 |
| LMF (CHM) | `locate_trees_lmf_chm` | CHM 栅格 (float64) | 树顶 (M,3) float64 = (x,y,z) |
| LMF (点云) | `locate_trees_lmf_points` | XYZ 点云 (N,3) float64 | 树顶 (M,3) float64 |
| Smooth height | `smooth_height` | XYZ 点云 + 邻域参数 | 平滑后的 z (N,) float64 |
| Watershed | `segment_watershed` | CHM 栅格 (float64) | 树冠 ID 栅格 (H,W) int32 |

CHM 与点云一律用 **float64**（与 lidR `NumericMatrix` / `double` 一致，便于 v0.2 fixture 对齐）。CHM 内存代价对典型尺寸（≤ 几千 × 几千）可忽略。

`watershed` 是**唯一不进入 C++ core 的算法**——上游 lidR 也仅是 `EBImage::watershed` 的 R 包装，没有 lidR 自有的 C++ 实现。pylidar 在 Python 层用 `skimage.segmentation.watershed` + `skimage.morphology.h_maxima` 实现，参数语义对齐 lidR 文档（`th_tree`、`tol`、`ext`）。

### 3.2 Out-of-scope（v0.1）

- DTM 插值、地面分类、点云度量、波形处理（架构留接口，目录暂不创建）
- LAS/LAZ 文件 IO（永远不做，用户用 `laspy`）
- ABI3 / stable Python ABI（v0.2 优化项）
- Python 端 Ctrl-C 软中断进 C++ 长循环（用户用 `signal` 模块自理）

## 4. 关键决策摘要

| 决策项 | 取值 |
|---|---|
| C++ 标准 | C++17（`std::span` 缺失，自实现 `Span<T>`） |
| 矩阵布局 | `Matrix2D<T>` **列主序**，与 lidR/R 一致 |
| 点云数据接口 | `PointCloudXYZ` POD（3 个 `const double*` + `size_t n`） |
| 栅格数据接口 | `RasterView<T>`：`Matrix2D<T>` + `(origin_x, origin_y, pixel_size)`（仅平移 + 各向同性像素尺度，**不支持旋转/剪切**） |
| 空间索引 | vendor 第三方 nanoflann（BSD-2，header-only），与上游 lidR 一致 |
| 并发 | 保留 OpenMP，承担 macOS libomp / MSVC `/openmp:llvm` / manylinux libgomp 三套打包成本 |
| Python 绑定 | nanobind ≥ 2.0 |
| Python 最低版本 | `>=3.10` |
| 构建后端 | scikit-build-core ≥ 0.10 |
| Wheel 矩阵 | cibuildwheel：Linux x86_64+aarch64（manylinux_2_28）/ macOS x86_64+arm64 / Windows x86_64，per-version wheel（3.10–3.14） |
| 错误处理 | 输入校验 Python 层做（`TypeError` / `ValueError`）；C++ 内部不变量违反抛 `std::invalid_argument` / `std::runtime_error`，nanobind 自动转 Python 异常 |
| 日志 | 全局可选回调 `pylidar.set_log_callback(callable)` → C++ 端 `std::function<void(std::string_view)>` |
| LAS/LAZ IO | 不做 |
| 测试基线（Day-1） | 合成数据语义自洽；与 lidR 数值对齐 fixture 通过独立 GH Actions workflow 离线生成提交 |
| License | GPL-3.0-or-later（继承 lidR） |

## 5. 仓库目录结构

```
pylidar/
├── pyproject.toml
├── CMakeLists.txt
├── LICENSE                              # GPL-3 全文
├── NOTICE                               # nanoflann BSD-2 attribution
├── README.md
├── src/
│   ├── core/                            # 纯 C++17，零 R/Python 依赖
│   │   ├── CMakeLists.txt               # 产 static lib `pylidar_core`
│   │   ├── third_party/
│   │   │   └── nanoflann.hpp            # vendored，BSD-2
│   │   ├── common/
│   │   │   ├── matrix2d.hpp             # 列主序 Matrix2D<T>
│   │   │   ├── point_cloud.hpp          # PointCloudXYZ + RasterView<T> + TreeTop POD
│   │   │   ├── span.hpp                 # 5 行 Span<T>，C++17 替代 std::span
│   │   │   ├── nanoflann_adaptor.hpp    # PointCloudXYZ → nanoflann KDTreeAdaptor
│   │   │   └── log.hpp                  # 全局 std::function<void(std::string_view)> 注册器
│   │   └── its/
│   │       ├── CMakeLists.txt
│   │       ├── dalponte2016.{hpp,cpp}
│   │       ├── silva2016.{hpp,cpp}
│   │       ├── li2012.{hpp,cpp}
│   │       ├── lmf.{hpp,cpp}
│   │       └── smooth_height.{hpp,cpp}
│   └── bindings/
│       ├── CMakeLists.txt               # 产 nanobind 扩展 `_core`
│       └── module.cpp                   # 全部 nb::module_ 定义集中此处
├── python/
│   └── pylidar/
│       ├── __init__.py                  # 公共 API re-export + __version__
│       ├── _core.pyi                    # nanobind 扩展类型存根
│       ├── segmentation.py              # 高层包装 + watershed（Python+skimage 实现）
│       └── _validate.py                 # 输入 dtype/shape/contiguity 校验
├── tests/
│   ├── conftest.py                      # 合成 fixture
│   ├── fixtures/                        # CI 离线生成的 .npz（v0.1 先空，v0.2 填）
│   ├── test_dalponte2016.py
│   ├── test_silva2016.py
│   ├── test_li2012.py
│   ├── test_lmf.py
│   ├── test_smooth_height.py
│   ├── test_watershed.py
│   ├── test_matrix2d.py                 # 列主序 / 越界 / view 语义
│   └── test_api_smoke.py                # 公共 API 形面冒烟
├── benches/
│   └── bench_*.py                       # pytest-benchmark
├── scripts/
│   └── generate_fixtures.R              # 由 GH Actions 调，跑 lidR 出参考输出
└── .github/workflows/
    ├── ci.yml                           # PR 触发：build+pytest（不含 R）
    ├── wheels.yml                       # tag 触发：cibuildwheel 多平台 wheel
    └── generate_fixtures.yml            # workflow_dispatch：装 R+lidR、跑脚本、提交回仓
```

`core/dtm/` `core/ground/` `core/metrics/` `core/waveform/` 等其他算法族目录在 v0.1 **不创建**（YAGNI）。但 `core/common/` 严格不依赖 `core/its/`，每个算法族独立 cmake target，确保未来加目录是零摩擦。

## 6. C++ core 设计

### 6.1 `pylidar::common` 类型

```cpp
namespace pylidar::common {

template<class T>
class Matrix2D {
public:
    Matrix2D(std::size_t rows, std::size_t cols);
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    T&       at(std::size_t r, std::size_t c) noexcept;
    const T& at(std::size_t r, std::size_t c) const noexcept;
    T*       data() noexcept;       // 列主序原始指针
    const T* data() const noexcept;
    // 不可拷贝赋值（防止意外复制大矩阵），可移动
private:
    std::size_t rows_, cols_;
    std::unique_ptr<T[]> buf_;      // 长度 rows_ * cols_，列主序：buf_[c*rows_ + r]
};

template<class T>
struct RasterView {
    Matrix2D<T>  data;              // 拥有 buffer
    double       origin_x;          // 像素 (row=0, col=0) 中心的世界 x
    double       origin_y;          // 像素 (row=0, col=0) 中心的世界 y
    double       pixel_size;        // 边长，> 0
};

struct PointCloudXYZ {              // 非拥有，POD
    const double* x;
    const double* y;
    const double* z;
    std::size_t   n;
    std::size_t   stride;           // 第 i 个点的坐标 = x[i*stride], y[i*stride], z[i*stride]
                                    // stride 单位 = double 元素数（不是字节）
                                    // (N,3) row-major numpy: x=&buf[0], y=&buf[1], z=&buf[2], stride=3
                                    // 独立 vector<double>: stride=1
};

struct TreeTop {
    double x, y, z;
    int    id;                      // 已分配的 ID；未分配时由调用方置 0
};

template<class T>
struct Span {                       // C++17 std::span 替代品
    T*           data;
    std::size_t  size;
    T*       begin() noexcept       { return data; }
    T*       end()   noexcept       { return data + size; }
    T&       operator[](std::size_t i) noexcept { return data[i]; }
};

}  // namespace pylidar::common
```

约定：
- 行 row 对应世界 y 反向（栅格顶行 = 北边）。具体公式：`world_x = origin_x + col * pixel_size`，`world_y = origin_y - row * pixel_size`。这与 GIS 通用栅格语义一致。
- `PointCloudXYZ` 的 stride 设计让 nanoflann adaptor 既能直接吃 `(N,3)` numpy 又能吃独立的 `vector<double>`。

### 6.2 nanoflann adaptor

```cpp
namespace pylidar::common {

struct PointCloudXYZ_KDAdaptor {
    const PointCloudXYZ& pc;
    inline std::size_t kdtree_get_point_count() const { return pc.n; }
    inline double kdtree_get_pt(std::size_t idx, std::size_t dim) const {
        const double* base = (dim == 0 ? pc.x : dim == 1 ? pc.y : pc.z);
        return base[idx * pc.stride];
    }
    template<class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree2D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudXYZ_KDAdaptor>,
    PointCloudXYZ_KDAdaptor, 2 /* dim, 仅 XY */>;

using KDTree3D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudXYZ_KDAdaptor>,
    PointCloudXYZ_KDAdaptor, 3>;

}
```

li2012 / lmf-points / smooth_height 内部各自 build 一棵 kd-tree（不缓存，每次调用重建）。日后若用户想跨算法复用索引，可在 v0.2 加 explicit `SpatialIndexHandle` 接口。

### 6.3 算法 API（`namespace pylidar::its`）

```cpp
namespace pylidar::its {

using common::Matrix2D;
using common::RasterView;
using common::PointCloudXYZ;
using common::TreeTop;

enum class Shape  { Circular, Square };
enum class SmoothMethod { Mean, Median, Gaussian };

// Dalponte 2016：CHM 区域增长。源 = src/C_dalponte2016.cpp 直译。
Matrix2D<int> dalponte2016(
    const RasterView<double>&    chm,
    const std::vector<TreeTop>&  seeds,       // 世界 XY，已排序（按 z 降序由调用方保证），id 已赋值
    double th_seed   = 0.45,
    double th_cr     = 0.55,
    double th_tree   = 2.0,
    double max_cr    = 10.0);

// Silva 2016：基于树顶 Voronoi + 距离/高度比阈值剔除。源 = R/algorithm-its.R::silva2016 翻译为 C++。
Matrix2D<int> silva2016(
    const RasterView<double>&    chm,
    const std::vector<TreeTop>&  seeds,       // id 已赋值
    double max_cr_factor = 0.6,
    double exclusion     = 0.3);

// Li 2012：基于点云的迭代分配。源 = LAS::segment_trees。
std::vector<int> li2012(
    const PointCloudXYZ& pts,
    double dt1      = 1.5,
    double dt2      = 2.0,
    double Zu       = 15.0,
    double R        = 2.0,
    double hmin     = 2.0,
    double speed_up = 10.0);

// LMF on raster：滑窗局部极大值。源 = LAS::fast_local_maximum_filter (raster path)。
// 输出 TreeTop 的 id 字段为 0；调用方（Python segmentation.py）按需要分配。
std::vector<TreeTop> lmf_chm(
    const RasterView<double>& chm,
    double ws,                  // 窗口直径或边长，世界单位
    double hmin,
    Shape  shape = Shape::Circular);

// LMF on points：球形/柱形邻域局部极大值。源 = LAS::fast_local_maximum_filter (points path)。
std::vector<TreeTop> lmf_points(
    const PointCloudXYZ& pts,
    double ws,
    double hmin,
    Shape  shape = Shape::Circular);

// Smooth height：对点云每点的 z 值在邻域内做平滑。源 = LAS::z_smooth。
std::vector<double> smooth_height(
    const PointCloudXYZ& pts,
    double size,
    SmoothMethod method,
    Shape  shape,
    double sigma = 0.0);  // 仅 Gaussian 用

}  // namespace pylidar::its
```

### 6.4 错误与日志

- 输入校验主战场在 Python `_validate.py`，C++ 仅做"内部不变量"检查（如 `chm.data.rows() > 0`、`pixel_size > 0`），违反时抛 `std::invalid_argument`。
- 算法运行期不可恢复错误（如 kd-tree 构建失败）抛 `std::runtime_error`。
- 删除 lidR 的 `Rcpp::checkUserInterrupt()`：Python 端用户自行用 `signal` 处理 Ctrl-C。
- 日志：`pylidar::common::log` 提供 `set_callback(std::function<void(std::string_view)>)` 与 `LIDAR_LOG(level, msg)` 宏。默认回调为 no-op。Python 端通过 `pylidar.set_log_callback(callable)` 注册。

## 7. Python 公共 API

公共名都从 `python/pylidar/__init__.py` re-export。

```python
import numpy as np
import pylidar

# CHM transform 约定：3-tuple (origin_x, origin_y, pixel_size)
#   origin = chm[0, 0] 像素中心的世界坐标
#   row 0 = 北边（最大 y），world_y = origin_y - row * pixel_size
#   pixel_size > 0，各向同性，不支持旋转/剪切

# 树顶检测 —— CHM 模式
ttops = pylidar.locate_trees_lmf_chm(
    chm,                               # (H, W) float64, row-major C-contiguous
    transform=(xmin, ymax, 0.5),
    ws=5.0,                            # 世界单位
    hmin=2.0,
    shape="circular",                  # "circular" | "square"
)
# → ndarray (M, 3) float64: columns = (x_world, y_world, z_max)

# 树顶检测 —— 点云模式
ttops = pylidar.locate_trees_lmf_points(
    xyz,                               # (N, 3) float64, row-major
    ws=5.0, hmin=2.0, shape="circular",
)

# 分割 —— Dalponte / Silva / Watershed（栅格输入栅格输出）
crowns = pylidar.segment_dalponte2016(
    chm, transform=(...), seeds=ttops,
    th_seed=0.45, th_cr=0.55, th_tree=2.0, max_cr=10.0,
)
# → ndarray (H, W) int32: 0 = no tree, ≥1 = tree id

crowns = pylidar.segment_silva2016(chm, transform=(...), seeds=ttops,
                                    max_cr_factor=0.6, exclusion=0.3)
crowns = pylidar.segment_watershed(chm, transform=(...),
                                    th_tree=2.0, tol=1.0, ext=1)

# 分割 —— Li2012（点云输入点云输出）
ids = pylidar.segment_li2012(
    xyz, dt1=1.5, dt2=2.0, Zu=15.0, R=2.0, hmin=2.0, speed_up=10.0,
)
# → ndarray (N,) int32

# 高度平滑
z_smooth = pylidar.smooth_height(
    xyz, size=2.0, method="mean", shape="circular", sigma=None,
)
# → ndarray (N,) float64

# 日志
pylidar.set_log_callback(print)        # 默认 None = 静默
```

**树顶 → seeds 的 ID 衔接**：`locate_trees_lmf_*` 返回的 `(M, 3)` 数组没有 ID 列。当传给 `segment_dalponte2016` / `segment_silva2016` 作为 `seeds` 时，Python 层在 `segmentation.py` 内自动按行号分配 `1..M` 作为默认 ID（因为 C++ 端 `dalponte2016/silva2016` 要求 `vector<TreeTop>` 的 `id` 已就绪）。如果用户想指定自定义 ID，可改传 `(M, 4)` 数组（最后一列 = int 转 float 的 id），bindings 层据此判断分支。

`segment_watershed` 实现位于 `python/pylidar/segmentation.py`，使用 `skimage.morphology.h_maxima`（对应 lidR 的 `tol`/`ext` 参数）+ `skimage.segmentation.watershed`（对应 `th_tree` mask）。

## 8. 数据通路与零拷贝

### 8.1 栅格

- 用户传入 `(H, W)` row-major C-contiguous numpy float64。
- bindings 层做 **一次** `O(H·W)` 转置拷贝到列主序 `Matrix2D<double>`：保 lidR 算法源码可对照、保 Python 用户友好。
- 输出 `Matrix2D<int>` 同样转置回 row-major numpy。
- 这是有意识的 trade-off。CHM 通常 ≤ 几十 MB，单次 memcpy 与算法本身耗时相比可忽略。

### 8.2 点云

- 用户传入 `(N, 3)` row-major C-contiguous numpy float64。
- bindings 层用 `nb::ndarray` 拿到底层指针，构造 `PointCloudXYZ{ x=&buf[0], y=&buf[1], z=&buf[2], n=N, stride=3 }`。**零拷贝**。
- 输出 `vector<int>` / `vector<double>` 通过 nanobind 直接构造为 numpy（一次小拷贝，O(N)）。

### 8.3 输入校验（Python `_validate.py`）

- dtype 必须精确（CHM = float64，点云 = float64，IDs = int32）；不匹配报 `TypeError` 并指引 `arr.astype(...)`
- shape 必须匹配（CHM 必须 2D；XYZ 必须 `(N, 3)`）；不匹配报 `ValueError`
- 必须 C-contiguous；非连续报 `ValueError` 并指引 `np.ascontiguousarray(arr)`
- NaN/Inf 检测可选（`validate=True` 关键字），默认 False（性能考虑）；NaN 在算法内的语义遵循 lidR：z=NaN 视作不可达
- `transform` 是 `(float, float, float)` 三元组，`pixel_size > 0`

## 9. 构建与发布

### 9.1 `pyproject.toml` 关键字段

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[project]
name = "pylidar"
version = "0.1.0"
requires-python = ">=3.10"
license = "GPL-3.0-or-later"
authors = [{ name = "TODO_AUTHOR_NAME", email = "TODO_AUTHOR_EMAIL" }]   # ← 实施时填
description = "Individual tree segmentation algorithms ported from R lidR"
readme = "README.md"
dependencies = [
    "numpy>=1.24",
    "scikit-image>=0.21",
]
classifiers = [
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: GIS",
]

[tool.scikit-build]
cmake.version = ">=3.26"
build-dir = "build/{wheel_tag}"
wheel.packages = ["python/pylidar"]
```

### 9.2 顶层 `CMakeLists.txt` 骨架

```cmake
cmake_minimum_required(VERSION 3.26)
project(pylidar LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(src/core)
add_subdirectory(src/bindings)
```

### 9.3 cibuildwheel 矩阵

```toml
[tool.cibuildwheel]
build = "cp310-* cp311-* cp312-* cp313-* cp314-*"
skip = "*-musllinux_*"   # v0.1 不支 musl，v0.2 加
test-requires = ["pytest", "scikit-image"]
test-command = "pytest {project}/tests -m 'not requires_fixture'"

[tool.cibuildwheel.linux]
manylinux-x86_64-image = "manylinux_2_28"
manylinux-aarch64-image = "manylinux_2_28"
archs = ["x86_64", "aarch64"]

[tool.cibuildwheel.macos]
archs = ["x86_64", "arm64"]
before-build = "brew install libomp"
environment = { LDFLAGS = "-L$(brew --prefix libomp)/lib", CPPFLAGS = "-I$(brew --prefix libomp)/include" }
repair-wheel-command = "delocate-wheel --require-archs {delocate_archs} -w {dest_dir} -v {wheel}"

[tool.cibuildwheel.windows]
archs = ["AMD64"]
environment = { CL = "/openmp:llvm" }
```

注：MSVC 默认 `/openmp` 是 OpenMP 2.0，不支 `#pragma omp atomic capture` 等现代特性；必须用 `/openmp:llvm`（LLVM 后端）或退而用 `/openmp:experimental`。

## 10. 测试策略

### 10.1 Day-1（合成数据语义自洽）

每个算法 ≥ 3 个 pytest case，覆盖：

- **已知答案**：手工构造小输入（5×5 CHM 含 3 个手放极大值；3 簇分离合成点云树）→ 断言算法输出与人工预期对齐（数量、位置、ID 计数）。
- **退化输入**：空数组、单点、单 NaN、全平地、全部 < hmin → 必须返回空 / 全 0，不可 segfault。
- **参数边界**：`hmin > max(z)`（应返回空）、`ws=0`（应抛 `ValueError`）、`max_cr` 极大（应不丢点）。

### 10.2 v0.2（与 lidR 对齐 fixture）

- `scripts/generate_fixtures.R`：跑 5 组合成数据 + 1 组 lidR 自带 `Megaplot.laz` 子集；每个算法的输入和参考输出存为 `tests/fixtures/<algo>/<case>.npz`。
- `.github/workflows/generate_fixtures.yml`：`workflow_dispatch` 触发；`r-lib/actions/setup-r@v2` 装 R 4.4；`Rscript -e 'install.packages("lidR")'`；运行 `Rscript scripts/generate_fixtures.R`；用 `peter-evans/create-pull-request` 把生成的 `.npz` 提 PR 回当前分支。
- pytest 用 `@pytest.mark.requires_fixture` 标记，`addopts = "-m 'not requires_fixture'"` 默认跳过；fixture 提交后 CI 自然就 cover。
- 容差：
  - dalponte / silva：每对树冠区域计算 IoU，整体 mean IoU > 0.95
  - lmf：树顶位置 KD-NN 配对，`> 95%` 配对距离 < `0.5 * pixel_size`
  - li2012：`sklearn.metrics.adjusted_rand_score(ids_pylidar, ids_lidr) > 0.95`（标签置换不变）
  - smooth_height：`np.allclose(z_pylidar, z_lidr, rtol=1e-6, atol=1e-9)`

### 10.3 性能基准（`benches/`）

`pytest-benchmark`，每个算法跑 3 个尺寸（小 / 中 / 大），与 lidR（fixture 生成时记录的耗时）对比。仅本地用，不进 CI gating。

## 11. 算法迁移映射

| pylidar 算法 | lidR v4.3.2 来源 | 迁移路径 |
|---|---|---|
| `dalponte2016` | `src/C_dalponte2016.cpp`（126 行） | 去 Rcpp 直译；`Rcpp::NumericMatrix` → `RasterView<double>`，`Rcpp::IntegerMatrix` 输出 → `Matrix2D<int>` |
| `silva2016` | `R/algorithm-its.R` 第 203-325（纯 R） | R→C++ 翻译；核心 = 树顶 Voronoi + `(distance / max_cr_factor) > exclusion * z` 阈值剔除 |
| `li2012` | `src/LAS.cpp::segment_trees`（在 1795 行 LAS 类内） | 抽算法主体；移除 LAS 类耦合（不依赖 LAS 的 IO/属性字段）；spatial index 用 `nanoflann` 替代 LAS 内嵌索引 |
| `lmf_chm` | `src/LAS.cpp::fast_local_maximum_filter` 的 raster 分支 | 抽出，吃 `RasterView<double>` |
| `lmf_points` | `src/LAS.cpp::fast_local_maximum_filter` 的 point cloud 分支 | 抽出，吃 `PointCloudXYZ` + nanoflann radius search |
| `smooth_height` | `src/LAS.cpp::z_smooth` | 抽出；mean/median 用 nanoflann radius；gaussian 用核宽 = `3σ` 截断 |
| `segment_watershed` | `R/algorithm-its.R::watershed`（wrap `EBImage::watershed`） | **不进 core**；Python 用 `skimage.morphology.h_maxima(chm, h=tol)` 找标记，再 `skimage.segmentation.watershed(-chm, markers, mask=chm>th_tree)`，外圈 dilate `ext` 像素 |

去 Rcpp 映射表（来自 rd.md）：

| Rcpp / R | C++ 等价 |
|---|---|
| `Rcpp::NumericMatrix` / `IntegerMatrix` | `Matrix2D<double>` / `Matrix2D<int>`，列主序 |
| `Rcpp::stop(...)` | `throw std::invalid_argument(...)` 或 `std::runtime_error(...)` |
| `Rcpp::Rcout` / `Rcerr` | 删除，或经 `pylidar::common::log` 回调 |
| `Rcpp::checkUserInterrupt()` | 删除 |
| `R_NaN` / `NA_REAL` | `std::numeric_limits<double>::quiet_NaN()` |
| `S4 las.slot("data")` | `const PointCloudXYZ&` |
| `Rcpp::List` 返回 | 自定义 POD struct（如 `TreeTop`） |
| `#pragma omp ...` | 保留，按需加 `#ifdef _OPENMP` 守护 |

## 12. 扩展点（v0.2+）

未来加 DTM / metrics / waveform / ground 时的预期形态：

```
src/core/
├── common/                              # 不动，保持只放跨算法族基础类型
├── its/                                 # 已有
├── dtm/                                 # 新增：dtm_kriging.cpp, dtm_idw.cpp, ...
├── metrics/                             # 新增：std_metrics.cpp, voxel_metrics.cpp
├── ground/                              # 新增：csf.cpp, pmf.cpp
└── waveform/                            # 新增
```

不变量（架构契约）：
- `core/common/` 不依赖任何 `core/<algo>/`
- 每个算法族在 `src/core/<algo>/CMakeLists.txt` 独立 target
- bindings 按算法族拆文件（v0.2 起 `module.cpp` → `module_its.cpp` + `module_dtm.cpp` + ...）
- Python 端按算法族提供 submodule（`pylidar.its`, `pylidar.dtm`, ...），但 v0.1 因为只有一族直接放顶层

## 13. 风险与未决

| 风险 | 缓解 |
|---|---|
| silva2016 R→C++ 翻译可能引入语义偏差 | v0.2 lidR fixture 对照（IoU > 0.95）兜底；翻译期严格按 `R/algorithm-its.R` 逐行 |
| li2012 从 LAS 类抽取时遗漏隐式状态 | 翻译前先用 R 跑几组输入记录中间状态，C++ 端逐 step 对照；OpenMP 关掉跑一次 deterministic 基线 |
| macOS libomp 在 cibuildwheel 上有平台特定坑 | manylinux 走 libgomp（已知稳）；macOS 用 `delocate-wheel` 标准 recipe；v0.1 第一次发版前手工 wheel 验装一次 |
| MSVC `/openmp:llvm` 与某些 OpenMP pragma 不兼容 | 算法内的 pragma 写最低集合（`parallel for` + `reduction`），不用 `task` / `atomic capture` |
| nanoflann ABI 在 header-only 升级时可能破坏 | vendor 锁定一个版本（v1.5.x），不跟主线；NOTICE 标版本 |
| GPL-3 与企业用户的兼容 | 在 README 顶部明示，鼓励有商用需求的用户单独商谈双许可（与 lidR 作者也是这个模式） |

未决（不阻塞 v0.1 实施）：

- nanobind 释放 GIL 的策略：`nb::call_guard<nb::gil_scoped_release>` 默认全开还是按算法选择？倾向**全开**（OpenMP 内部不持 GIL），但要测一次 log 回调期间的死锁可能（log 回调若回到 Python 需要重获 GIL）。
- `transform` 的类型：3-tuple 还是 dataclass `RasterTransform(origin_x, origin_y, pixel_size)`？v0.1 用 3-tuple（最简），v0.2 视用户反馈引 dataclass。
- 整型 ID 类型：v0.1 用 int32，足够。v0.2 若有用户跑超过 21 亿点的场景再加 int64 路径。
- 自定义 seed ID 的 API 形态：本文 §7 用 "(M,4) 数组带 id 列" 兼容自定义 ID，但用户也可能想直接传 dict 或带名字的 record array；v0.1 先做 (M,3) / (M,4) 两路，复杂的 v0.2 视反馈再扩展。

## 14. 附录 A：lidR 各算法参数语义

仅列出 v0.1 涉及的参数，详见 `R/algorithm-its.R` 与 lidR 文档。

### Dalponte 2016
- `th_tree`：CHM 像素值低于此视为非树（mask）
- `th_seed`：邻域增长时，候选像素值 ≥ `th_seed * crown_seed_z` 才被接受
- `th_cr`：候选像素值 ≥ `th_cr * crown_mean_z` 才被接受（更严格）
- `max_cr`：单棵树冠半径上限（世界单位）

### Silva 2016
- `max_cr_factor`：树冠半径估计 = `max_cr_factor * tree_height`
- `exclusion`：像素到任一树顶的高度差 / 距离 比若超过此阈值，则该像素被剔除（不属于任何树）

### Li 2012
- `dt1` / `dt2`：阈值距离的两个尺度
- `Zu`：z 阈值，区分使用 dt1 还是 dt2 的高度边界
- `R`：搜索半径
- `hmin`：低于此高度的点不参与
- `speed_up`：邻域剪枝距离上限

### LMF
- `ws`：滑窗（CHM）或邻域（点云）大小（世界单位，直径或边长）
- `hmin`：低于此高度的极大值候选被丢弃
- `shape`：圆形或方形邻域

### Smooth height
- `size`：邻域大小（直径或边长）
- `method`：mean / median / gaussian
- `sigma`：高斯核 σ（仅 gaussian）

### Watershed
- `th_tree`：低于此高度的像素被 mask
- `tol`：`h_maxima` 的高度差阈值（合并相邻局部极大值）
- `ext`：watershed 完成后向外膨胀的像素数

## 15. 参考

- lidR GitHub：https://github.com/r-lidar/lidR
- 本地 lidR 仓库：`/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR`（v4.3.2）
- nanoflann：https://github.com/jlblancoc/nanoflann（BSD-2，header-only）
- nanobind：https://github.com/wjakob/nanobind
- scikit-build-core：https://scikit-build-core.readthedocs.io
- cibuildwheel：https://cibuildwheel.readthedocs.io
- 项目内需求文档：`rd.md`（注：rd.md 中的部分文件路径基于 lidR 旧版本，v4.3.2 的实际位置见本设计第 11 节）
