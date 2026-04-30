# Findings & Decisions — pylidar v0.1

## Requirements (来自 `rd.md` + 6 轮 brainstorm 对齐)

- 把 lidR 的 6 个个体树分割算法（dalponte2016 / silva2016 / li2012 / lmf-chm / lmf-points / smooth_height）+ watershed 迁移到 Python 可 pip 安装的包
- 算法核心：纯 C++17，零 R/Rcpp/Python 依赖（`core/` 子树可独立链接到任意 C++ 项目）
- Python 入口：nanobind 绑定 + 高层 `pylidar.segmentation` 模块
- 构建：scikit-build-core + cibuildwheel 多平台 wheel
- 架构按算法族分目录（`core/its/` / 未来 `core/dtm/` / `core/metrics/` ...），不平铺
- LAS/LAZ 文件 IO 永远在 pylidar 之外（用户用 laspy）
- License = GPL-3-or-later（继承 lidR）

## Research Findings — lidR 源码现状（v4.3.2）

**关键发现（2026-04-30 探查）**：rd.md "范围"表里的若干文件路径基于旧版 lidR，与当前 v4.3.2 不一致。权威映射：

| rd.md 引用 | 实际位置 | 性质 |
|---|---|---|
| `C_dalponte2016.cpp` | `src/C_dalponte2016.cpp`（126 行） | ✅ 仍是独立 C++，可直接迁移 |
| `C_silva2016.cpp` | **不存在** | silva2016 是**纯 R 实现**（`R/algorithm-its.R` 第 203-325，~120 行） |
| `C_li2012.cpp` + `SpatialIndex.cpp` + `Point.h` | **不存在** | li2012 现位于 `LAS::segment_trees` 方法（嵌入 1795 行 `LAS.cpp`） |
| `C_lmf.cpp` | `RcppFunction.cpp` 第 37 行 → `LAS::fast_local_maximum_filter` | 同时支持 raster 和 point cloud 两种输入 |
| `C_smooth.cpp` | `RcppFunction.cpp` 第 45 行 → `LAS::z_smooth` | 注意是对**点云的 Z 值**做平滑，**不是** CHM |
| `watershed` C++ | **不存在** | `R/algorithm-its.R` 第 328 行直接 wrap `EBImage::watershed` |
| 空间索引 | `src/nanoflann/` 第三方 vendored（BSD-2 单 header） | 已被 lidR 自身使用 |

实质工作量（修正后）：
- 直接 C++ 迁移：dalponte2016 / lmf / smooth_height
- 从 LAS 类抽取算法：li2012（最复杂）
- R→C++ 翻译：silva2016（高风险）
- 完全不进 C++ core：watershed（Python 层 wrap skimage）

## Technical Decisions（由 brainstorm 6 轮对话锁定，权威来源 = `docs/specs/2026-04-30-pylidar-its-design.md`）

| Decision | Rationale |
|----------|-----------|
| C++ 标准 = C++17 | rd.md 既定；自实现 5 行 `Span<T>` 替代 C++20 std::span |
| `Matrix2D<T>` 列主序 | 与 lidR/R `NumericMatrix` 一致，便于源码直译 + fixture 容差对齐 |
| CHM/点云全 float64 | 对齐 lidR `double`；CHM 体积小（≤ 几十 MB），无内存压力 |
| `RasterView<T>`：data + (origin_x, origin_y, pixel_size) | 仅平移 + 各向同性像素尺度，**不**支持旋转/剪切 |
| `PointCloudXYZ`：3 个 `const double*` + n + stride（POD，非拥有） | 支持 (N,3) row-major numpy 零拷贝 + 独立 vector 两种用法 |
| Vendor nanoflann v1.5.x（BSD-2 单 header） | 与上游 lidR 一致；GPL-3 兼容；NOTICE attribution |
| 保留 OpenMP | 性能必要；接受 macOS libomp + MSVC `/openmp:llvm` + manylinux libgomp 三套打包成本 |
| Bindings = nanobind ≥ 2.0 | 编译快、wheel 小、C++17 入门刚好对齐 |
| Python `>=3.10`，per-version wheel | numpy 2.x 下限 3.10；ABI3 限制太多不值（v0.2 优化） |
| LMF 暴露两个独立 Python 函数（`locate_trees_lmf_chm` / `locate_trees_lmf_points`） | "Explicit is better than implicit"；避免 shape 二义性分发 |
| `transform = (origin_x, origin_y, pixel_size)` 3-tuple | v0.1 最简；v0.2 视用户反馈引 dataclass |
| Watershed 不进 C++ core | 上游 lidR 也仅 wrap EBImage；Python 层用 `skimage.morphology.h_maxima` + `skimage.segmentation.watershed` 边界更干净 |
| 测试 Day-1 走合成（C 思路） | 用户本地无 R 环境；不阻塞日常开发 |
| Fixture（B 思路）通过独立 GH Actions workflow 离线生成 | 装 R 仅在 fixture 生成 workflow，不污染日常 PR CI |
| 错误：Python 层校验抛 TypeError/ValueError；C++ 内部抛 std::invalid_argument/runtime_error | nanobind 自动转换；用户拿到 Python 风格 traceback |
| 日志：全局 `set_log_callback`；C++ 端 `std::function<void(std::string_view)>` | rd.md 既定 |
| 栅格 row-major numpy → 列主序 Matrix2D 在 bindings 层一次 `O(H·W)` 转置拷贝 | 有意识 trade-off：保 lidR 源码可对照 + Python 用户友好；CHM 通常 ≤ 几十 MB，单次 memcpy 可忽略 |
| 点云 (N,3) row-major numpy → PointCloudXYZ 零拷贝（用 stride=3 切片） | nanobind ndarray 直接拿底层指针 |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| rd.md 文件清单基于旧版 lidR | brainstorm 第 1 轮即向用户披露，spec §11 给权威映射表 |
| writing-plans skill 不存在 | 改用 planning-with-files skill 出 task_plan.md / findings.md / progress.md（本套文件） |

## Resources

- **Design spec（权威）**：`docs/specs/2026-04-30-pylidar-its-design.md`
- **需求**：`rd.md`（注：部分文件路径过时，以 spec §11 为准）
- **上游 lidR**：`/Users/lambdayin/Code-Projects/maicro_projects/3d/third_party/lidR`（v4.3.2，本机已 clone）
- **lidR GitHub**：https://github.com/r-lidar/lidR
- **关键源文件**：
  - `src/C_dalponte2016.cpp`（126 行，dalponte 算法）
  - `R/algorithm-its.R`（517 行，含 silva2016 / dalponte2016 R 包装 / watershed / li2012 R 包装）
  - `src/LAS.cpp`（1795 行，含 `segment_trees` / `fast_local_maximum_filter` / `z_smooth`）
  - `src/RcppFunction.cpp`（500 行，C_lmf / C_smooth / C_li2012 入口）
  - `R/locate_trees.R` + `R/locate_localmaxima.R`（lmf R 包装）
  - `src/nanoflann/`（已 vendored）
- nanoflann GitHub（用于 vendor）：https://github.com/jlblancoc/nanoflann（v1.5.x）
- nanobind：https://github.com/wjakob/nanobind
- scikit-build-core：https://scikit-build-core.readthedocs.io
- cibuildwheel：https://cibuildwheel.readthedocs.io
- skimage watershed：`skimage.morphology.h_maxima` + `skimage.segmentation.watershed`

## Visual/Browser Findings

无（本任务全文本，未涉及图像/PDF/浏览器）。

## Open Items（不阻塞 v0.1 实施，记录追踪）

- nanobind GIL 释放策略：默认全开 `nb::call_guard<nb::gil_scoped_release>`，但 log 回调回 Python 时需重获 GIL → Phase 1 验证
- `transform` v0.2 是否引 dataclass `RasterTransform(origin_x, origin_y, pixel_size)`
- v0.2 是否加 int64 ID 路径（>21 亿点场景）
- 自定义 seed ID API（v0.1 用 (M,4) 数组带 id 列；v0.2 视反馈是否引 dict / record array）
- 跨算法复用 kd-tree 的 `SpatialIndexHandle`（v0.2 优化项，v0.1 每次调用重建）

## Risks（spec §13，实施时关注）

- silva2016 R→C++ 翻译可能引入语义偏差 → Phase 4 mid-phase 用户审翻译笔记 + v0.2 fixture 容差兜底
- li2012 从 LAS 类抽取时遗漏隐式状态 → Phase 5 启动前先读 LAS.cpp 写依赖图
- macOS libomp / MSVC `/openmp:llvm` / manylinux libgomp 三套打包路径 → Phase 7 单独 phase 处理
- nanoflann ABI 漂移 → vendor 锁版本 v1.5.x，NOTICE 标版本，不跟主线
- GPL-3 与企业用户兼容 → README 顶部明示，鼓励双许可商谈
