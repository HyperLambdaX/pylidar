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

## Recurring Lessons（每次 phase 开始前扫一眼）

> 从 phase 内的 debug 经验里挑出"下一个 phase 极可能再撞"的几条。冷启动（`/clear` 后）回到这里，比从头读代码更快。

### nanobind 2.x — 绑定写法的几个坑

- **不要用 `std::optional<nb::callable>` 或 `nb::callable` 作参数类型。** 实测 nb 2.x 在 Python 端传 `builtin_function_or_method`（如 `list.append`）时不会匹配进去，会抛 `TypeError: incompatible function arguments`。代替方案：参数用 `nb::object`，函数体里 `PyCallable_Check(obj.ptr())` 自己校验。Phase 0 的 `set_log_callback` 就是这么改的；之后任何接收回调的 binding 都按这个套路写。
- **`nb::python_error::discard_as_unraisable` 是 non-const 成员函数。** `catch (const nb::python_error&)` 编译不过；要写 `catch (nb::python_error& e)`。
- **GIL 边界：** 算法函数加 `nb::call_guard<nb::gil_scoped_release>()` 就能释放 GIL，但只要回调回到 Python（log、用户传入的 functor）就必须 `nb::gil_scoped_acquire` 重获。`set_log_callback` 的 lambda 已经示范这个 pattern。
- **`@stl/function.h` / `@stl/optional.h` 的隐式转换不万能。** 一旦绑定签名被 nanobind 拒绝，第一反应是回到 `nb::object` + 手动 narrow，不要在 STL 转换上反复试。
- **绑定写错时编译期未必报错，运行时才会暴露。** 任何新 `m.def(...)` 必须配 pytest smoke，不能只靠"编译通过"。

### 构建链路

- **macOS Apple Clang 的 OpenMP shim 已在顶层 `CMakeLists.txt` 里。** 如果 Phase 1+ 引入 `#pragma omp` 后链接失败，问题大概率不在 shim 本身（`-Xpreprocessor -fopenmp -lomp` 已配好），而在算法 static lib 没继承 `OpenMP::OpenMP_CXX`。届时给 `pylidar_its` 加 `target_link_libraries(... PUBLIC OpenMP::OpenMP_CXX)`。
- **scikit-build-core ≥ 0.12 严格执行 PEP 639。** SPDX `license = "..."` 与旧式 `License :: OSI Approved :: ...` classifier **不能共存**，构建会直接 fail。任何时候改 `pyproject.toml` 都别加 License classifier。
- **本地 dev 装包流程（macOS arm64 已验，uv-native）：**
  ```sh
  # 首次：创建 venv + 装 editable wheel + test 依赖
  uv venv --python 3.14
  uv pip install -e ".[test]"

  # 跑测试——不需要 source .venv/bin/activate，uv run 自己挑当前项目的 env
  uv run pytest tests -m "not requires_fixture"
  ```
  增量改 C++ 后只需 `uv pip install -e ".[test]" --no-deps` 重装（不再 resolve 依赖，~3s），然后 `uv run pytest ...`。`uv run` 本身不会触发 CMake 重建——所有 C++ 编译路径都走 `uv pip install -e`。Python-only 改动则可以直接 `uv run pytest`，省一步装包。
- **`wheel.packages = ["python/pylidar"]` + `install(TARGETS _core LIBRARY DESTINATION pylidar)`** 的搭配让 nanobind 扩展落进 `pylidar/` 包目录。Phase 1+ 如果再加扩展或多 wheel target，确认 `DESTINATION` 仍是 `pylidar`，不要写成 `pylidar/_core/`。

### 测试与验证习惯

- **每个新 binding 同一 PR 内必须有 pytest smoke。** Phase 0 的 4 个 smoke 是模板：import → 扩展 load → 函数可调 → 错误类型可触发。Phase 1+ 加算法函数照抄。
- **`pyproject.toml` 里 `addopts = "-m 'not requires_fixture'"` 默认跳过 lidR-fixture 测试**，所以本地不装 R 也能跑。Phase 8 加 `@pytest.mark.requires_fixture` 标记的对照测试就靠这个 marker。

### nanoflann 用法（Phase 1 抽出）

- **L2_Simple_Adaptor 的默认 IndexType 是 `uint32_t`**，不是 `size_t`。`std::vector<nanoflann::ResultItem<std::uint32_t, double>> matches;` 是匹配 KDTree2D/3D 默认实例化的结果类型。写错会编译失败但报错很晦涩。
- **radius 用平方距离**：`tree.radiusSearch(query, radius_sq, matches)` 第二参是 `radius * radius`（L2 metric 的语义）。文档 §1723 注释明确写过——抄错会少／多匹配 sqrt(2) 倍邻居。
- **adaptor 不支持 bbox query**。需要矩形/方形邻域时的 pattern：用外接圆半径（square 边长 `s` → `radius = s*sqrt(2)/2` → `radius_sq = s²/2`）做 radius search，再在循环里用 `abs(dx)<=half_size && abs(dy)<=half_size` 过滤。`smooth_height.cpp` 的 Square 分支是模板。
- **`SearchParameters{}.sorted = false`** 显式关掉排序——我们要的是邻居集合而不是 nearest-K，没必要付那个 sort 成本。

### C++17 可移植性细节（Phase 1 抽出）

- **MSVC 默认不带 `M_PI`**（要 `_USE_MATH_DEFINES`）。**用 `std::acos(-1.0)`** 在所有平台一行解决。
- **OpenMP parallel for 的循环变量用 `std::ptrdiff_t`/有符号整型**，不要用 `size_t`。Apple Clang/GCC 都接受 unsigned，但 MSVC OpenMP 2.0 不行；为了 CI 三平台都过，统一有符号。
- **算法库要直接 link `OpenMP::OpenMP_CXX`**，不能只靠顶层 `find_package`。Phase 1 改 `pylidar_its` 从 INTERFACE 升级到 STATIC 时同时把 OpenMP 加到 PUBLIC link——遗漏会导致 `#pragma omp parallel for` 在该 .cpp 里被静默忽略，单线程跑过测试，性能问题晚一截才暴露。

### lmf 算法的几个端口注意点（Phase 2 抽出）

- **lidR 的 raster-LMF 不是独立算法。** `locate_trees(<SpatRaster>, lmf(...))` 走的是 `R/locate_trees.R:106-148` 的 dispatch，先把 CHM 用 `raster_as_las()` 转成 1-point-per-cell 的伪 LAS，然后调同一个 `C_lmf`。pylidar 的 `lmf_chm` 在 C++ 端复刻这条路径——构造虚拟点云投给 `lmf_filter_impl`，与 `lmf_points` 共算法主体。**含义：** 不要去手写 raster sweep 风格的实现（即便它在 dense CHM 上更快）；统一走 kd-tree 路径才能 v0.2 fixture 对照仅做一次。
- **从上游 LMF 主动偏离了两处（lmf.cpp 文件头有同样总结）：**
  1. **Tie-break 改成确定性 `z == zi && j < i`。** lidR 内层 `if (z == zmax && filter[pt.id])` 是裸读全局 `filter`，与并行写无锁同步。两个邻居 z 完全相等时，输出取决于线程调度——本地实测 4 个等高点同窗，重复跑能拿到 1 / 4 / "最后一个" 三种结果。改成"最低索引赢"既单线程多线程一致，也与一次串行跑的结果完全相同。**不要**回到 race-y 版本以求"fixture parity"——lidR 自己的输出在那个 corner 里也是 non-deterministic 的，不存在能对齐的"参考值"。回归测试见 `test_lmf_points_tied_heights_are_deterministic`。
  2. **删了 `state[j] = kNLM` 跨迭代写优化。** 上游靠它给"短邻居"打永久-NLM 标记好让外层迭代直接跳过。Tie-break 改确定性后这条写没算法价值，且本身是又一次跨迭代裸写。删掉以后每个外层迭代只写自己的 `filter[i_u]`（不同索引、并行安全），整个 parallel-for 完全 data-race-free，可以同时把 `#pragma omp critical` 去掉，并在见到 `z > zi` 时立即 `break`（之前不能 break 是因为还要继续打 NLM 标记）。
- **`zmax` 在新版本里也删了。** lidR 的 `zmax = Z[i]` 永不更新，唯一用处是 race-y tie-break 的左操作数；我们既然换规则就直接去掉这个冗余变量，可读性更好。
- **C++ 必须自己校验 hmin。** spec §6.4 明确：core/ 可被独立 link，bypass Python wrapper 时 hmin=NaN 会让 `zi >= hmin` 永为 false → 静默返回空集，与 Phase 1 size/sigma 同类陷阱。lmf 加了 `if (!std::isfinite(hmin)) throw std::invalid_argument(...)`。**模板：每个新算法的 C++ 入口对所有数值参数都要 `isfinite` 校验**，不能只信 Python wrapper。回归测试见 `test_lmf_core_rejects_nonfinite_hmin_directly[nan/inf/-inf]`。
- **`if (!(zi >= hmin)) continue;` 同时挡 NaN z。** 双重否定让 NaN 也被排除（`NaN >= x` 是 false），无须额外 `isnan` 检查。算法里所有"过滤掉非有限值"的地方都用这个 idiom。
- **CHM 像素坐标到世界坐标的约定。** `RasterView::world_x(c) = origin_x + c * pixel_size`，`world_y(r) = origin_y - r * pixel_size`（GIS 北上：row=0 是最大 y）。test_lmf_chm_three_peaks_transform_world_xy 是 reference。dalponte/silva 抽时记得复用 `RasterView` 这两 helper。

### Phase 2 binding 模板（Phase 3+ 复用）

- **(M, 3) 输出的标准 helper：** `module.cpp::treetops_to_numpy_xyz(std::vector<TreeTop>&&)` 已就绪，把 TreeTop POD 拷成 row-major (M,3) float64 numpy。下一阶段 dalponte/silva 接收 `seeds` 输入也走 `nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig>`（自定义 ID 时 (M,4)）。
- **CHM 输入模板：** `nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>` + 三个独立 double（origin_x, origin_y, pixel_size）。binding 函数体内一次 `H·W` 嵌套循环复制到列主序 `Matrix2D<double>`。dalponte/silva 直接抄这段。
- **`check_shape_int(int)` 已抽为共享。** 其他算法接收 shape 参数时复用，避免重复 `if (shape != 1 && shape != 2)`。
- **`_validate.py::ensure_chm_float64` / `ensure_transform`** 已就绪可复用。

### lidR 上游的小 bug（Phase 1 撞到，纯供 fixture 生成时绕开）

- `R/smooth_height.R:45` —— `if (method == "circle") shape <- 1 else shape <- 2`，本意应为 `if (shape == "circle") shape <- 1 else shape <- 2`。结果 R API 的 `shape="square"` 实际走 `C_smooth(..., shape=2, ...)` 即 Circle。
- 影响：Phase 8 生成 smooth_height 的 lidR 参考 fixture 时**不要**走 R 包装；直接 `lidR:::C_smooth(las, size, method, shape=1, sigma, ncpu)`（C 入口，shape=1 强制 Rectangle）才能拿到真正的 square 输出对照 pylidar。circular 走 R API 没问题。
- 我们 C++ 端的 Square / Circular 行为不受影响——已经从 `LAS::z_smooth` 那一层直译，没经过 R 包装那行。

### 输入校验的"NaN 静默通过"陷阱（Phase 1 抽出）

- **`x <= 0` 不能挡 NaN**。`float('nan') <= 0` 是 `False`，`std::isnan(x) && x <= 0` 也是 `false`。NaN 会一路穿到算法里，下游 nanoflann 的 `dist < NaN` 又是 false，结果是"零邻居 → 用回退分支输出原始 z 值，零警告"。
- **正数参数的 idiomatic check**：Python 用 `not math.isfinite(x) or x <= 0`；C++ 用 `!std::isfinite(x) || x <= 0.0`。**inf 也要拦**——`half_res = inf` 同样把搜索半径推到非正常区。
- **双层都查**。Spec §6.4 写明 Python 层负责 user-facing 校验，C++ 内部不变量也要查。理由：下游可能直接 link `pylidar::core` 或调 `_core.*` 绕过 wrapper（我们的 NaN 测试就是这么测 C++ 层的）。
- **测试模板**：每个数值参数加 `pytest.mark.parametrize("bad", [nan, inf, -inf])`，外加一个 `_core.<algo>(..., nan, ...)` 的直调测试覆盖 C++ 层。Phase 1 `tests/test_smooth_height.py` 末尾 3 个 test 是模板。

### nanobind 异常映射（Phase 1 实测）

- `std::invalid_argument` → Python `ValueError`（自动）
- `std::out_of_range` → `IndexError`
- `std::runtime_error` → `RuntimeError`
- 算法内部不变量违反就抛 `std::invalid_argument`，不要自己手动 `throw nb::value_error(...)`——保持 C++ core 与 binding 解耦，core 不依赖 nanobind 头。

### 全局回调的重入安全（Phase 1 抽出）

- **`std::function` 的全局存储 + `emit() { const auto& cb = storage(); cb(); }` 是雷**：如果 callback 内部触发 `set_callback({})`，存储被 move，回调中的 `cb` 引用就悬了。
- **修法**：emit 里把 callback **拷到栈上** 再调（`Callback cb = storage(); if (cb) cb(...);`）。我们的 callback 内含 `shared_ptr<nb::object>`，拷贝 = 一次原子 ref-count 增量，非热路径开销可忽略。
- **跨线程并发改 callback** 仍是 caller 责任（不是 emit() 能 fix 的）。文档里讲清楚，别在算法运行期间动 callback。

### Span<T> 默认状态的 UB（Phase 1 修复）

- **`data + size` 在 `data=nullptr, size=0` 时是 UB**（C++17 [expr.add]：指针算术要求 P 指向数组元素或末尾，nullptr 不指向任何数组）。GCC/Clang/MSVC 实践上不出怪事，但 `-fsanitize=undefined` 会标红。
- **修法**：`end()` 写 `return size ? data + size : data;`——空时直接返 `data`，避免对空指针做加法。`begin()` 不需要改（直接返 `data` 没算术）。
- 后续算法用 `Span<T>{}` 表示空入参时不会被这条坑到。

### Python 包装层模式（Phase 1 抽出）

- **C++ 算法函数返回 `std::vector<double>` → bindings 里手动转 numpy**（`new double[n]` + `nb::capsule` 拥有 + `nb::ndarray<nb::numpy, double, nb::ndim<1>>`）。比 `@stl/vector.h` 给出来的 Python list 体面，比直接用 `nb::ndarray` 写 in-place 输出参数干净。模板已在 `module.cpp::vector_to_numpy_1d` 里。
- **GIL 用 `nb::gil_scoped_release` 块**（不是 `call_guard`）包住对 C++ 算法的调用——这样异常 / 返回值的 Python 类型构造仍持 GIL 执行，只有真正 CPU 重活在无 GIL 期间。`nb::call_guard<nb::gil_scoped_release>()` 也行但 release 范围更宽，遇到 nanobind 内部要 GIL 的边界场景更难调。
- **Python 层默认值就地 fallback** 比让 C++ 处理 `optional<double>` 简单太多。`segmentation.py::smooth_height` 的 `sigma is None → size/6` 是模板；不要把这逻辑塞进 binding。
