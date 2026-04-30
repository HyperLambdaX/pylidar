# Task Plan: pylidar v0.1 — ITS 算法迁移实施

## Goal

把 R 包 lidR (v4.3.2) 的 6 个 ITS 算法迁移成纯 C++17 内核 + nanobind Python 绑定 + 可 `pip install` 的多平台 wheel；架构按算法族分目录，为后续迁移 DTM/metrics/waveform 留扩展点。

## Current Phase

Phase 1 — smooth_height（完成，待 commit / Phase 2 启动）

## Reference Documents

- **Design spec**：`docs/specs/2026-04-30-pylidar-its-design.md`（**所有架构决策的最终来源**，本计划任何一处与 spec 冲突以 spec 为准）
- **需求**：`rd.md`（注：rd.md 中部分文件路径基于旧版 lidR，权威映射见 spec §11）
- **上游 lidR**：`/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR`（v4.3.2）
- **研究笔记**：`findings.md`
- **会话日志**：`progress.md`

## Phases

### Phase 0: 项目骨架（不含算法）

**目标**：搭好"可 `pip install -e .` + `pytest` 跑空骨架测试"的最小架子。所有架构问题（CMake glue、nanobind 链路、cibuildwheel 雏形、目录布局）在此暴露并解决。

- [x] 重写 `pyproject.toml`（scikit-build-core、nanobind、`requires-python>=3.10`、GPL-3-or-later、numpy/skimage 依赖）
- [x] 创建 `LICENSE`（GPL-3 全文）、`NOTICE`（nanoflann BSD-2 attribution）
- [x] 顶层 `CMakeLists.txt`：find OpenMP / nanobind，add_subdirectory core/bindings
- [x] `src/core/CMakeLists.txt` + `src/core/its/CMakeLists.txt`（占位 target）
- [x] `src/bindings/CMakeLists.txt`
- [x] 下载 vendor `src/core/third_party/nanoflann.hpp`（v1.5.5，单 header）
- [x] `src/core/common/`：`matrix2d.hpp`（列主序）、`point_cloud.hpp`（POD + TreeTop）、`span.hpp`、`log.hpp`、`nanoflann_adaptor.hpp`
- [x] `src/bindings/module.cpp`：仅注册 `set_log_callback`（验证 binding 链路）
- [x] `python/pylidar/__init__.py`、`_core.pyi`、`_validate.py`、`segmentation.py` 骨架
- [x] `tests/test_matrix2d.py`（v0.1 占位，C++-only 类型；Phase 1+ 替换）、`tests/test_api_smoke.py`
- [x] `.github/workflows/ci.yml`（matrix Linux/macOS/Windows × Py 3.10/3.12/3.14）
- [x] **Acceptance**：本机 `uv pip install -e ".[test]"` 成功；`pytest tests -m "not requires_fixture"` 4 passed / 1 skipped；CI 待 push 后验证
- **Status:** complete

### Phase 1: smooth_height（最简算法，验证全链路）

**目标**：完整走通"Python 校验 → bindings → C++ 算法 → kd-tree → 返回 numpy"。架构问题如有遗漏在这一步暴露。

- [x] `src/core/its/smooth_height.{hpp,cpp}`（从 `LAS.cpp::z_smooth` 抽取）
- [x] `src/bindings/module.cpp` 加 `_core.smooth_height`
- [x] `python/pylidar/segmentation.py` 实现 `smooth_height(...)` 包装
- [x] `python/pylidar/__init__.py` re-export
- [x] `python/pylidar/_core.pyi` 补存根
- [x] `tests/test_smooth_height.py`（实交 6 case：4 task_plan 必需 + 2 额外校验：错 dtype / 错 shape 字符串）
- [x] **Acceptance**：本机 `pytest tests -m "not requires_fixture"` = 10 passed, 1 skipped（Phase 0 占位 Matrix2D 仍 skip）
- **Status:** complete

### Phase 2: lmf_chm + lmf_points

- [ ] `src/core/its/lmf.{hpp,cpp}`（抽 `LAS::fast_local_maximum_filter` 两分支）
- [ ] bindings 加 `_core.lmf_chm` 和 `_core.lmf_points`
- [ ] segmentation.py 实现 `locate_trees_lmf_chm` / `locate_trees_lmf_points`
- [ ] `tests/test_lmf.py`（5 case：5×5 CHM 三峰 / hmin 过滤 / 三簇点云 / ws=0 抛 ValueError / CHM 全 NaN）
- [ ] **Acceptance**：5 测试全过；transform 把 row/col 转世界 XY 手算对照
- **Status:** pending

### Phase 3: dalponte2016

- [ ] `src/core/its/dalponte2016.{hpp,cpp}`（从 `src/C_dalponte2016.cpp` 126 行直译，去 Rcpp）
- [ ] bindings + segmentation.py（含 seeds 形态 (M,3) / (M,4) 自动 ID 分配）
- [ ] `tests/test_dalponte2016.py`（5 case：单峰单 seed / 双峰双 seed / seed 在 mask 区域 / 自定义 ID / dtype 错抛 TypeError）
- [ ] **Acceptance**：5 测试全过
- **Status:** pending

### Phase 4: silva2016（**R→C++ 翻译，最高风险 phase**）

- [ ] 先写 `docs/notes/silva2016-translation-trace.md`（工作笔记，行号对照 R/algorithm-its.R 第 203-325，gitignored）
- [ ] **GATE：用户审阅翻译笔记后再提交代码**（mid-phase user review，因风险高）
- [ ] `src/core/its/silva2016.{hpp,cpp}`
- [ ] bindings + segmentation.py
- [ ] `tests/test_silva2016.py`（5 case：双树对比 dalponte 更保守 / 高度差大 exclusion 生效 / max_cr_factor 极大 / 单 seed / exclusion=0）
- [ ] **Acceptance**：5 测试全过 + 翻译笔记被用户 ack
- **Status:** pending

### Phase 5: li2012（**最复杂，从 1795 行 LAS.cpp 抽算法**）

- [ ] 先读 `LAS.cpp` 整文件，标注 `segment_trees` 依赖的私有字段/方法 → 写 `findings.md` 的 "li2012 LAS dependency map"
- [ ] `src/core/its/li2012.{hpp,cpp}`：抽算法主体，nanoflann KDTree2D 替代 LAS 内嵌索引
- [ ] bindings + segmentation.py
- [ ] `tests/test_li2012.py`（5 case：3 簇分离 → 3 ID / 单树 / 1 主树+离群点 / 全 z<hmin → 全 0 / 1000 点压力测试）
- [ ] **Acceptance**：5 测试全过；在合成数据上无 NaN/segfault
- **Status:** pending

### Phase 6: watershed（Python 层，无 C++）

- [ ] `python/pylidar/segmentation.py` 加 `segment_watershed`，使用 `skimage.morphology.h_maxima(chm, h=tol)` + `skimage.segmentation.watershed(-chm, markers, mask=chm>th_tree)` + 外圈 dilate `ext` 像素
- [ ] `tests/test_watershed.py`（3 case：双峰 / 单峰 / 全平地）
- [ ] **Acceptance**：3 测试全过
- **Status:** pending

### Phase 7: cibuildwheel 多平台 wheel

- [ ] `.github/workflows/wheels.yml`（tag 触发 `v*`）
- [ ] `pyproject.toml` 加 `[tool.cibuildwheel.*]` 段（macOS libomp + delocate-wheel；Windows `/openmp:llvm`；Linux manylinux_2_28）
- [ ] 第一次发版前本机 `cibuildwheel --platform linux` 跑一次手验
- [ ] **Acceptance**：手动 dispatch workflow，三平台 wheel 都构建成功 + 你本机 `pip install <wheel>` 可装可 import
- **Status:** pending

### Phase 8: lidR fixture 离线生成 + v0.2 容差对照

- [ ] `scripts/generate_fixtures.R`（5 组合成 + 1 组 Megaplot 子集，产 `.npz`）
- [ ] `.github/workflows/generate_fixtures.yml`（`workflow_dispatch`，装 R+lidR，跑脚本，create-pull-request 提 PR）
- [ ] phase 1-6 已有 pytest 文件加 `@pytest.mark.requires_fixture` 的对照测试
- [ ] `pytest.ini` 加 markers + 默认 `addopts = "-m 'not requires_fixture'"`
- [ ] **Acceptance**：手动触发 workflow → 得到 PR → merge 后 `pytest tests -m requires_fixture` 全过
- **Status:** pending

## Key Questions

1. nanobind 的 `nb::call_guard<nb::gil_scoped_release>` 是否对每个算法函数都开？默认全开（OpenMP 不持 GIL），但 log 回调回 Python 时需重获 GIL → Phase 1 验证。
2. `transform` 的 Python 表示 v0.1 用 3-tuple，v0.2 是否引 dataclass？看 Phase 2 用户反馈。
3. li2012 从 LAS.cpp 抽取时哪些 LAS 私有字段是必须的？Phase 5 启动前先读 LAS.cpp 写依赖图。
4. silva2016 R→C++ 翻译的逐行对照笔记是否充分？Phase 4 mid-phase gate。

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| C++17，自实现 5 行 `Span<T>` 替代 std::span | rd.md 定 C++17；std::span 是 C++20；不升级避免 toolchain 兼容问题 |
| `Matrix2D<T>` 列主序 | 与 lidR/R `NumericMatrix` 一致，便于 lidR 源码直译与 fixture 对齐 |
| CHM/点云一律 float64 | 对齐 lidR `double`；CHM 体积小（≤几十 MB），无内存压力 |
| Vendor nanoflann（BSD-2 单 header） | 与上游 lidR 一致；GPL-3 兼容；无 ABI 风险 |
| 保留 OpenMP | 性能必要；接受 macOS libomp + MSVC `/openmp:llvm` + manylinux libgomp 三套打包成本 |
| Bindings = nanobind ≥ 2.0 | 编译快、wheel 小、C++17 入门刚好对齐 |
| Python `>=3.10`，per-version wheel（不走 ABI3） | numpy 2.x 下限 3.10；ABI3 限制太多不值（v0.2 优化） |
| Watershed 不进 C++ core | 上游 lidR 也仅 wrap EBImage；Python 层用 skimage 边界更干净 |
| 测试 Day-1 走合成（C 思路），fixture（B 思路）通过 GH Actions 离线生成 | 用户本地无 R 环境；CI 装 R 不影响日常 PR CI |
| 错误：Python 层校验抛 TypeError/ValueError；C++ 不变量违反抛 std::invalid_argument/runtime_error | nanobind 自动转换；用户拿到 Python 风格 traceback |
| 提交策略：每 phase 一个 commit；spec 已写但未提交（按 rd.md "重大设计需先问"约定） | 用户决定何时 commit |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Notes

- **Spec 是权威**：本文件任何冲突以 `docs/specs/2026-04-30-pylidar-its-design.md` 为准
- **每 phase 完成后**：更新 Status，progress.md 记录会话，新发现写 findings.md
- **Phase 间严格串行**：每个 phase 都依赖前一个的产出（Phase 0 的 common 类型，Phase 1 的 binding 链路验证 等）
- **Phase 4 / 5 风险高**：silva2016 翻译需 mid-phase 用户审；li2012 抽取前需先写 LAS 依赖图
- **未提交的资产**：`docs/specs/2026-04-30-pylidar-its-design.md`（用户决定何时 commit）、本计划三文件
- 任何"重大决策"（多文件影响 / API 改动 / 新依赖）按 rd.md 约定先问用户
