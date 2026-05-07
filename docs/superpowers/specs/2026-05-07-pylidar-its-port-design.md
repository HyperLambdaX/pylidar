# pylidar — lidR ITS 算法迁移 · 执行设计 (Spec)

- 日期：2026-05-07
- 来源需求：`rd.md`（与本文同级仓库根目录）
- 上游参考：lidR (https://github.com/r-lidar/lidR)，本地镜像 `/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR`
- License：GPL-3.0-or-later（沿袭 lidR）

---

## 1. 范围与目标

把 lidR 中**单木分割（Individual Tree Segmentation, ITS）**相关算法迁移到 Python，作为后续多算法族迁移的起点。本期：

- 算法落地范围：**P0~P2 全部**——`dalponte2016` / `silva2016` / `li2012` / `lmf` / `chm_smooth` / `watershed`
- 交付形态：本机用 §5 "Dev 模式" 三步装上、`from pylidar.segmentation import ...` 可调
- 平台：M0~M3 验收以 **macOS (Apple Silicon)** 为准；**Linux (Ubuntu) 跑通延后到 M3 完成之后**单独补，CMake 不放任何 macOS-only 路径
- **不在本期**：PyPI 发布、跨平台 wheel、CI、Windows、OpenMP 并行（rd.md §6）、KD-tree 加速 li2012（rd.md §7）

## 2. 里程碑（M0 → M3）

每个里程碑都是**独立可验证、独立 commit**。§5 "Dev 模式" 三步配方 + `uv run pytest -q` 必须全绿。每个里程碑完成后**停下汇报**，等用户决定再开下一个。

### M0 · 脚手架（无算法）
- `pyproject.toml`、顶层 + 子目录 `CMakeLists.txt`、`LICENSE` (GPL-3 全文)、`README.md`（一段说明 + 安装命令）、`.gitignore`
- `src/core/common/{point.hpp, matrix2d.hpp, kdtree.hpp}`：M0 仅 header 骨架。`kdtree.hpp` 仅包含 `#pragma once` + 一行"M2 will wrap nanoflann"注释，不引入 nanoflann（M2 才 vendor）
- `src/core/its/_placeholder.cpp`：让 `pylidar_core_its` 静态库能 link，M1 加入第一个真算法时删除
- `src/bindings/module.cpp`：仅暴露 `_core.__version__` 与 `_core.ping() -> str`
- `python/pylidar/__init__.py`：暴露 `__version__`
- `python/pylidar/_core.pyi`：手动维护，进 git
- 验证点（macOS）：§5 "Dev 模式" 三步成功；`uv run python -c "import pylidar; print(pylidar.__version__); import pylidar._core; print(pylidar._core.ping())"` 不报错；`uv run pytest -q` 全绿（M0 仅一个 smoke test 模块，覆盖以上 import + ping）

### M1 · P0（dalponte2016 + silva2016）
- `src/core/its/dalponte2016.{hpp,cpp}`：从 `lidR/src/C_dalponte2016.cpp` 1:1 翻译；输入 `Matrix2D<double>` CHM + `Matrix2D<int32_t>` seeds + 4 个阈值；输出 `Matrix2D<int32_t>` regions
- `src/bindings/module.cpp`：注册 `dalponte2016(...)`，dtype/contig 严格按 §4
- `python/pylidar/segmentation.py`：`silva2016(...)` 纯 Python（scipy.cKDTree + numpy）
- `tests/fixtures/`：`dalponte2016_*.npz`、`silva2016_*.npz`，每算法 ≥3 组（happy / 退化 / 行为对齐 corner），每文件 ≤50KB
- `tests/test_dalponte2016.py`、`tests/test_silva2016.py`
- 验证点：`uv run pytest -q` 全绿；整数标签输出（dalponte2016 regions、silva2016 ids）与 fixture 严格相等（`np.array_equal`）。本里程碑无浮点输出；§10.4 的 `np.allclose(rtol=1e-6, atol=0)` 容差适用于后续里程碑（如 chm_smooth）。

### M2 · P1（li2012 + lmf）
- `src/third_party/nanoflann/nanoflann.h` vendor（保留原 LICENSE）
- `src/core/common/kdtree.hpp` 填实（薄包装 nanoflann）
- `src/core/its/li2012.{hpp,cpp}`：从 `lidR/src/RcppFunction.cpp:92` + `lidR/src/LAS.cpp:1113` 翻译；**首版 1:1 行为对齐 lidR，含 O(N²) 距离扫描，不调 KD-tree**（rd.md §7）
- `src/core/its/lmf.{hpp,cpp}`：两个重载源（点云 / CHM）→ 拆为 `lmf_points` / `lmf_chm` 两个 C++ 函数；点云入口用 KD-tree
- bindings 注册：`li2012` / `lmf_points` / `lmf_chm`
- 测试 + fixture 跟进
- 备注：KD-tree 加速 `li2012` 是 M2 完成后的**独立性能 PR**，不在本里程碑

### M3 · P2（chm_smooth + watershed）
- `src/core/its/chm_smooth.{hpp,cpp}`：从 `lidR/src/RcppFunction.cpp:45` + `lidR/src/LAS.cpp:112` 翻译
- `python/pylidar/segmentation.py` 加 `watershed(...)`（skimage 包装）
- bindings 注册 `chm_smooth`
- 测试 + fixture 跟进

### 后 M3
- Ubuntu 平台跑通验证（CMake 不应需要修改）
- 可选：KD-tree 加速 `li2012`、性能基线写入 `tests/perf_baseline.json`
- 可选：用真 R+lidR 重生成 fixture（`tools/regen_fixtures.R`），`meta/source` 切到 `"lidR_run"`

## 3. 公开 API

**总规则**：
- 函数命名**完全照搬 lidR**（`dalponte2016` / `li2012` / `silva2016` / `chm_smooth` / `watershed`）
- 所有参数走 keyword（`*` 强制 keyword-only）
- 输入只接 numpy；不复刻 LAS S4
- 顶层 `pylidar` 不 re-export 算法，必须 `from pylidar.segmentation import ...`
- 顶层只暴露 `pylidar.__version__`

**签名**（最终形态，本期分里程碑增量实现）：

```python
# pylidar/segmentation.py 暴露
def dalponte2016(*, chm, seeds, th_tree=2.0, th_seed=0.45, th_cr=0.55, max_cr=10.0): ...
# chm: (H,W) float64 c_contig；seeds: (H,W) int32 c_contig；返回 (H,W) int32

def silva2016(*, xyz, treetops, max_cr_factor=0.6, exclusion=0.3): ...
# xyz: (N,3) float64；treetops: (M,3) float64；返回 (N,) int32

def li2012(*, xyz, dt1=1.5, dt2=2.0, R=2.0, Zu=15.0, hmin=2.0, speed_up=10.0): ...
# xyz: (N,3) float64；返回 (N,) int32

def lmf_points(*, xyz, ws, hmin=2.0, shape="circular"): ...
# xyz: (N,3) float64；ws: float 或 callable(z)->float；返回 (N,) bool
# 说明：callable→array 的展开在 bindings 层完成（对每个点的 z 调用一次），
# core 层只接 (N,) float64 ws 数组，不持有 Python 函数指针。

def lmf_chm(*, chm, ws, hmin=2.0, shape="circular"): ...
# chm: (H,W) float64；返回 (K,2) int32（行/列像素索引）

def chm_smooth(*, xyz, size=3, method="average", shape="circular"): ...
# xyz: (N,3) float64；返回 (N,) float64

def watershed(*, chm, th_tree=2.0, tol=1.0): ...
# chm: (H,W) float64；返回 (H,W) int32
```

**错误约定**：core 层抛 `std::invalid_argument` / `std::runtime_error`，nanobind 默认映射到 `ValueError` / `RuntimeError`，bindings 层不再二次包装。

**预条件检查的归属**：在 **bindings 层入口**做 dtype / shape / contig / 跨参数一致性检查；core 层假定输入合法、不重复检查（避免 hot loop 里塞断言）。

## 4. 仓库结构

M0 commit 后仓库形态：

```
pylidar/
├── pyproject.toml
├── CMakeLists.txt
├── LICENSE                            # GPL-3 全文
├── README.md
├── .gitignore
├── .python-version                    # 3.14
├── CLAUDE.md
├── rd.md
├── docs/superpowers/specs/
│   └── 2026-05-07-pylidar-its-port-design.md   # 本文
├── src/
│   ├── core/
│   │   ├── CMakeLists.txt             # add_subdirectory(common) / its
│   │   ├── common/
│   │   │   ├── CMakeLists.txt         # INTERFACE library: pylidar_core_common
│   │   │   ├── point.hpp              # PointXYZ POD
│   │   │   ├── matrix2d.hpp           # 行主序，header-only
│   │   │   └── kdtree.hpp             # M0 空 header；M2 包装 nanoflann
│   │   └── its/
│   │       ├── CMakeLists.txt         # STATIC library: pylidar_core_its
│   │       ├── _placeholder.cpp       # M0 占位；M1 删
│   │       ├── dalponte2016.{hpp,cpp} # M1
│   │       ├── li2012.{hpp,cpp}       # M2
│   │       ├── lmf.{hpp,cpp}          # M2
│   │       └── chm_smooth.{hpp,cpp}   # M3
│   ├── third_party/                   # M2 起 vendor
│   │   ├── README.md                  # 列每个 vendored lib 的版本+license
│   │   └── nanoflann/
│   │       ├── nanoflann.h
│   │       └── LICENSE
│   └── bindings/
│       ├── CMakeLists.txt             # nanobind_add_module(_core ...)
│       └── module.cpp                 # M0 起，每里程碑增量增函数
├── python/pylidar/
│   ├── __init__.py                    # 暴露 __version__
│   ├── _core.pyi                      # type stub，进 git
│   └── segmentation.py                # M1 起填（silva2016 / watershed 等纯 Python）
├── tests/
│   ├── conftest.py                    # load_fixture() 等共享工具
│   ├── fixtures/
│   │   ├── README.md                  # 每个 .npz 来源/参数说明
│   │   ├── dalponte2016_*.npz         # M1
│   │   ├── silva2016_*.npz            # M1
│   │   ├── li2012_*.npz               # M2
│   │   ├── lmf_*.npz                  # M2
│   │   ├── chm_smooth_*.npz           # M3
│   │   └── watershed_*.npz            # M3
│   ├── test_dalponte2016.py           # M1
│   ├── test_silva2016.py              # M1
│   ├── test_li2012.py                 # M2
│   ├── test_lmf.py                    # M2
│   ├── test_chm_smooth.py             # M3
│   └── test_watershed.py              # M3
└── tools/
    └── regen_fixtures.R               # 留档脚本（即使本期 fixture 是手推的也写）
```

**未来扩展约定**：新增算法族 = 加一个 `src/core/<family>/CMakeLists.txt` + 顶层 `add_subdirectory` + `python/pylidar/<family>.py`，**零回改既有目标**。

## 5. 构建与 CMake 拓扑

### `pyproject.toml`（M0 定型）

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.4"]
build-backend = "scikit_build_core.build"

[project]
name = "pylidar"
version = "0.1.0"
description = "Python port of lidR's individual tree segmentation algorithms"
readme = "README.md"
requires-python = ">=3.14"
license = { text = "GPL-3.0-or-later" }
dependencies = [
    "numpy>=2.1",
    "laspy>=2.5",
    "scipy>=1.13",
    "scikit-image>=0.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-benchmark>=4",
    # Build toolchain pinned in the venv so `--no-build-isolation` succeeds and
    # `editable.rebuild = true` can re-invoke CMake on .cpp/.hpp edits without
    # spinning up an ephemeral PEP-517 build env (whose ninja path would not
    # survive the original build, breaking the rebuild hook).
    "scikit-build-core>=0.10",
    "nanobind>=2.4",
    "ninja>=1.11",
]

[tool.scikit-build]
# {state} 分隔 editable / wheel / sdist 各自的构建目录，避免 wheel/use 模式
# (uv build --wheel / uv pip install .) 把临时 ninja 路径写入共享的
# CMakeCache.txt，导致下次 editable rebuild 失败（详见 §5 末段）。
build-dir = "build/{state}/{wheel_tag}"
editable.rebuild = true
editable.mode = "redirect"
cmake.version = ">=3.26"
cmake.build-type = "Release"
wheel.packages = ["python/pylidar"]
```

### CMake 顶层

```cmake
cmake_minimum_required(VERSION 3.26)
project(pylidar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# 注意：不开 -Werror（nanobind 模板宏会触发部分 -Wpedantic 警告）
# 注意：不开 -ffast-math（破坏 NaN 语义）

find_package(OpenMP COMPONENTS CXX QUIET)   # 找但不强求；本期源码 0 条 #pragma omp
find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(src/core)
add_subdirectory(src/bindings)
```

### `src/core/common/CMakeLists.txt`

```cmake
add_library(pylidar_core_common INTERFACE)
target_include_directories(pylidar_core_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(pylidar_core_common INTERFACE cxx_std_17)
```

### `src/core/its/CMakeLists.txt`（M0 形态）

```cmake
add_library(pylidar_core_its STATIC
    _placeholder.cpp
    # dalponte2016.cpp        # M1 取消注释
    # li2012.cpp              # M2
    # lmf.cpp                 # M2
    # chm_smooth.cpp          # M3
)
target_link_libraries(pylidar_core_its PUBLIC pylidar_core_common)
target_include_directories(pylidar_core_its PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
  target_link_libraries(pylidar_core_its PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(pylidar_core_its PRIVATE PYLIDAR_HAS_OPENMP=1)
endif()
```

### `src/bindings/CMakeLists.txt`

```cmake
nanobind_add_module(_core
    NB_STATIC
    STABLE_ABI
    module.cpp
)
target_link_libraries(_core PRIVATE pylidar_core_its)
install(TARGETS _core LIBRARY DESTINATION pylidar)
```

**单一 .so 产物**：nanobind 静态进 `_core`，`pylidar_core_*` 也是静态库——运行时只产出一个 `python/pylidar/_core.<abi>.so`，零 RPATH 折腾。

### 本地开发流程

项目全程用 [uv](https://docs.astral.sh/uv/) 管理，分两种模式：

#### Dev 模式（editable，编辑 .cpp / .hpp 自动 ninja 增量 rebuild）

```bash
# 1. 灌依赖+backend+ninja，但跳过 uv 自己装项目（uv 用 ~/.cache/uv/builds-v0/.tmp*
#    临时构建项目，会把短命 ninja 路径烤进 CMake cache，下次 import 触发
#    rebuild 时找不到 ninja → 失败）。
uv sync --extra dev --no-install-project

# 2. uv pip 装 editable，配 --no-build-isolation 让构建复用 venv 内稳定的
#    backend + ninja 路径；CMake cache 烤进 .venv/bin/ninja，editable.rebuild
#    钩子常驻可用。
uv pip install --no-build-isolation -e ".[dev]"

# 3. 跑测试。
uv run pytest -q
```

#### Use 模式（一次性装，不监听源码变化）

```bash
uv venv
uv pip install .
```

#### 关于 "uv 对 scikit-build-core editable rebuild 钩子支持弱" 的修订

rd.md 当时的判断是对**带 build-isolation** 的 `uv pip install -e` / `uv sync` 装项目的情形——uv 在 `~/.cache/uv/builds-v0/.tmp*/bin/ninja` 临时目录构建，CMake cache 烤死临时 ninja 路径，下次 import 时该路径已被 uv 清除，rebuild 钩子炸掉。配 `--no-build-isolation` 强制 uv 复用 venv 内的 backend + ninja，CMake 走稳定路径，鉴权钩子完全工作。本节 Dev 模式即此修订路径。

## 6. 测试与 fixture 策略

### fixture 文件格式

每个 `.npz` 包含 `inputs/*` + `expected/*` + `meta/*`。`meta/source` 字符串值固定一组：
- `"manual_derivation"`：从 lidR 源码人工推导（本期默认）
- `"lidR_run"`：真跑 lidR 得到（本期没有）

`meta/source` 必须包含 lidR 源文件路径 + 行号范围，例：`"manual_derivation: lidR/src/C_dalponte2016.cpp:1-180"`。

### 每算法至少三组 fixture

| 类别 | 例 |
|---|---|
| happy path | 10×10 CHM 含 3 个清晰 peak；li2012 50 点云 2 簇 |
| 退化输入 | 全零 CHM、单像素 CHM、空点云、单点点云、所有点同 z |
| 行为对齐 corner | dalponte2016 邻接区域共享边界 tie-break；li2012 距离恰好等于 dt1 |

每文件 ≤50KB，CHM ≤32×32，点云 ≤200 点。

### 测试组织

```python
# tests/conftest.py
import numpy as np
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"

def load_fixture(name: str) -> dict:
    return dict(np.load(FIXTURE_DIR / f"{name}.npz"))
```

每个 `test_<algo>.py` 按 happy / 退化 / corner 三类各覆盖至少一个用例。

### 性能基准

`pytest-benchmark` 记录数字到 `tests/perf_baseline.json`，**不卡 PR**、**不纳入 CI**，本期保持轻量。

### "我手推 fixture" 的诚实门槛

为避免"用算法实现去校准 fixture，再用同一份 fixture 校验算法实现"的循环依赖，**强制顺序**：

1. **先**纯人工读 lidR 源码，用纸笔/REPL 推一组小例子的输出
2. **再**把手算结果存入 `.npz`
3. **再**跑翻译实现，对比
4. 不一致时**默认实现错**（rd.md §7：1:1 行为对齐 lidR），调实现而不是调 fixture

## 7. 去 Rcpp 化执行细则

rd.md §3 给出主映射表，本节补全易踩的隐性约束。

### 索引与布局

- **0-based**：core 层全部 0-based。`Matrix2D<T>::operator()(row, col)` 0-based。lidR cpp 里的 1-based 索引（来自 R 习惯）全部转换。
- **行主序**：`Matrix2D<T>` 行主序，与 numpy 一致。lidR `NumericMatrix` 是列主序，但 ITS 算法对布局无偏好（flood-fill / 点距离），逐元素翻译循环即可，**不需要 transpose**。

### NaN 处理

- 用 `std::isnan(x)`，**不**写 `x != x`
- `pylidar_core_its` **不**加 `-ffast-math`（破坏 NaN 语义）

### 进度/日志

- core 层**不接收** `verbose` 参数
- bindings 层本期**不实现**进度回调；亚秒级算法不需要
- `Rcpp::Rcout` / `Rcerr` / `Rcpp::checkUserInterrupt()` 全部删除（rd.md §3）

### 多返回值

`Rcpp::List` 返回 → core 层 POD 结构体；bindings 层一次性 copy 转 ndarray 暴露给 Python。例：

```cpp
struct LmfChmResult {
    std::vector<int32_t> rows;
    std::vector<int32_t> cols;
};
```

### 异常归属

- core 层**只用异常**（`std::invalid_argument` / `std::runtime_error`）
- 预条件校验在 bindings 层入口；core 层假定输入合法
- 后果：未来如果有人在 C++ 层直接复用 core 函数，**需要自己校验**（这是有意的设计取舍）

## 8. PORT NOTE 模板

每个 C++ 翻译文件顶部都带这块：

```cpp
// PORT NOTE — porting from lidR
// Source:        <relative path inside lidR repo>:<line range>
//                e.g. lidR/src/C_dalponte2016.cpp:1-180
// lidR commit:   <hash if known, else "TBD: regen with R env">
// Layout:        lidR uses column-major NumericMatrix; we use row-major Matrix2D<T>.
//                The algorithm is layout-agnostic (flood-fill / point distance),
//                so loops are translated index-for-index. No transpose needed.
// Indexing:      0-based throughout. (lidR mixes 0-based C++ loops with 1-based
//                R-facing indices; all R-side conversions are dropped.)
// NaN:           std::isnan() guard. Do NOT enable -ffast-math.
// Threading:     Single-threaded (rd.md §6). No #pragma omp.
// Behavior:      1:1 with lidR (rd.md §7); intentional perf shortcuts deferred.
```

## 9. 沟通节奏

| 节点 | 我（Claude）的动作 | 用户的角色 |
|---|---|---|
| 写完 spec | git commit + 请用户 review spec | 通过/打回 |
| **每个里程碑（M0/M1/M2/M3）完成** | 单独 commit，**停下**总结，不自动开下一个；下一个里程碑由用户触发后再调 writing-plans 出新 plan | 决定继续/调整/暂停 |
| 翻译过程遇到 lidR 源码看不懂 | 直接读 GitHub 源码而不是猜（rd.md §沟通约定） | 不打扰 |
| 翻译时发现"明显是 lidR bug" | **停下**，给用户看 diff + 建议（回灌上游 vs 本地修） | 决定 |
| 单文件内部小决策（命名/注释/局部重构） | 自己定，不打断（rd.md §沟通约定） | 不打扰 |

## 10. 验收

### 本期硬性验收（macOS）

1. §5 "Dev 模式" 三步配方成功（`uv sync --extra dev --no-install-project` → `uv pip install --no-build-isolation -e ".[dev]"` → `uv run pytest -q`），无平台特异编译错误
2. `uv run python -c "import pylidar; from pylidar.segmentation import dalponte2016, silva2016, li2012, lmf_points, lmf_chm, chm_smooth, watershed"` 不报错
3. `uv run pytest -q` 全绿
4. **所有 6 个算法**端到端测试通过：浮点输出与 `tests/fixtures/` 中 `meta/source = "manual_derivation"` 的参考结果在 `np.allclose(rtol=1e-6, atol=0)` 容差内一致；整数标签输出（`dalponte2016` 的 regions、`li2012` / `silva2016` 的 tree id）严格相等。**rd.md §验收 §3 仅硬性要求 P0**，本 spec 把容差对齐推广到 P1/P2 是因为本期范围扩到 P2，统一标准

### 与 rd.md §验收 §3 的差异（已与用户确认）

rd.md 原文要求"输出与 **lidR fixture** 在 `np.allclose(rtol=1e-6)` 容差内一致"。本期 fixture 是从 lidR 源码人工推导的预期输出，不是 lidR 真跑结果。所以验收 §10.4 改写为"对齐 spec 内的 fixture（来源：manual_derivation）"。

**当未来有可用 R+lidR 环境时**：跑 `tools/regen_fixtures.R` 把 `.npz` 替换为 `meta/source = "lidR_run"` 的真值，并通过同一组测试——这是已埋的回归门，**不是本期门槛**。

### 后 M3（不阻塞本期）

- Ubuntu 平台 §5 "Dev 模式" 三步配方 + `uv run pytest -q` 跑通
- KD-tree 加速 `li2012`（独立 PR）
- 用真 lidR 重生 fixture

## 11. 依赖明细

| 依赖 | 版本下限 | 用途 | License |
|---|---|---|---|
| Python | 3.14 | runtime | PSF |
| numpy | 2.1 | 核心数据类型 | BSD-3 |
| laspy | 2.5 | LAS/LAZ 文件读取（用户侧） | BSD-2 |
| scipy | 1.13 | silva2016 KDTree | BSD-3 |
| scikit-image | 0.24 | watershed 包装 | BSD-3 |
| scikit-build-core | 0.10 | 构建后端 | Apache-2 |
| nanobind | 2.4 | C++/Python 绑定（构建期/dev） | BSD-3 |
| ninja | 1.11 | scikit-build-core editable rebuild 钩子运行时驱动（dev） | Apache-2 |
| nanoflann | latest stable | KD-tree（M2 vendor，header-only） | BSD-2 |
| pytest | 8 | 测试（dev） | MIT |
| pytest-benchmark | 4 | 性能记录（dev） | BSD-2 |

所有依赖与 GPL-3 兼容。

## 12. 已知 TBD

- 里程碑 commit 通过 `git tag m0..m3` 标记（每个里程碑收尾 commit 上打一个轻量 tag，便于回溯）
- lidR commit hash：本期 fixture 是手推；当真有 R 环境后填到各 `.npz` 的 `meta/lidR_commit_ref`
- Linux 验收：M3 完成后单独跑通

## 13. 不在本期的事项（提前记一笔避免架构失血）

- `point_metrics` / 特征值分解（依赖 lidR 的 `BH` + `RcppArmadillo`）→ 未来 `src/core/common/linalg/` 引入 Eigen 替代
- 多平台 wheel / PyPI 发布（项目名 `pylidar` 在 PyPI 已被占用，发布前需改名，候选：`pylidR` / `lidR-py` / `treesegpy`）
- OpenMP 并行（CMake 留口子，源码 0 条 `#pragma omp`）
- LAS S4 对象（永远不复刻，输入只接 numpy）
