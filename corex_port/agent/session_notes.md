# Flux → Iluvatar CoreX (ivcore11) 迁移记录

## 来源
- 仓库：bytedance/flux（https://github.com/bytedance/flux）
- 分支/commit：`main` @ `19831ca2d820e3e782ed1d15d8b52d0898b78b26`（2025-08-28）
- 含子模块：`3rdparty/cutlass`（CUTLASS 3.9.0）、`3rdparty/nccl`（NVIDIA NCCL），均已 `git submodule update --init` 拉取。

## CUDA 使用性质
Flux 是通信-计算重叠的 GPU kernel 库（dense/MoE 模型的 AllGather-GEMM、GEMM-ReduceScatter、MoE scatter/gather 等），**完全构建在 CUTLASS 3.9 之上**：
- 源码统计：`Sm90` 引用 759 处、`Sm80` 538 处、`Sm89` 26 处；大量 `wgmma`/`TMA`/`cp.async` 内建。
- 目录里全是 `sm80_*`、`sm90_*`、`*_tma_warpspecialized_*` kernel；tuning config 仅面向 A100(sm80)、H800/H20(sm90)。
- **无任何 Volta/sm70 代码路径**。库的设计核心就是 Hopper 的 TMA + warp-specialized cooperative kernel。
- 依赖 CUTLASS、NCCL、（可选）NVSHMEM、PyTorch，面向多卡。

## 环境
- `source /home/init_container.sh` 配好 CoreX 环境：clang/clang++ 22.1.0（4.5.0.20260630）、`ixsmi`、`GITHUB_TOKEN`、`GITHUB_ORG=tlmn2008`。
- GPU：2× Iluvatar BI-V150（各 32GiB），`ixsmi` 正常可见。
- CoreX toolkit 报 CUDA 10.2；`torch` 为 CoreX 版 2.10.0（`torch.cuda.is_available()=True`，含 FP8）。
- `nvcc` 是 Bourne-Again shell 脚本（stub，非真编译器）——按 SOP 一律用 clang `-x ivcore --cuda-gpu-arch=ivcore11`，禁用 nvcc。
- 重要发现：**CoreX 自带 nccl 与 nvshmem**（`/usr/local/corex/lib64/libnccl*.so`、`libnvshmem_host.so` + `include/nccl.h`、`nvshmem.h`），无需从源码用 nvcc 编 NCCL，可直接映射到 CoreX 通信库。

## 适配内容（逐步解决 workaround-able 障碍）
1. **CUDA>=11.0 门禁**：`CMakeLists.txt` 新增 `FLUX_COREX` 探测分支，检测到 `/usr/local/corex` 时跳过 `>=11.0` 硬 `FATAL_ERROR`，改走 ivcore11。
2. **nvcc-only 编译选项翻译**（参考 `iluvatar-cuda-base/nvcc-flag-translation`）：删除 `-gencode`、`--expt-extended-lambda`、`--expt-relaxed-constexpr`（clang 原生支持）；去掉 `-Xcompiler` 包装、`-Xcompiler=` 连写；跳过 Hopper 专用 `sm_90a` 改写；`-rdc=true` 经 `CCC_OVERRIDE_OPTIONS="s/-rdc=true/-fgpu-rdc/"` 在 driver 层替换。
3. **torch 2.10 API 漂移**：`cpp_extension._prepare_ldflags` 新增 `with_sycl`/`is_standalone` 参数，原 CMake 硬编码参数个数报 `TypeError`；改为按 `inspect.signature` 动态构造。
4. **CUTLASS 缺完整 libcu++/CCCL**：CoreX `cuda/std` 只有 9 项（无 `utility`/`array`/`tuple`），CUTLASS 3.9 `fast_math.h` 即 `#include <cuda/std/utility>` fatal。`pip install nvidia-cuda-cccl-cu12`(12.9.27) 补齐，经环境变量 `FLUX_COREX_CCCL_INCLUDE` 注入到主工程与 host 侧 kernel 生成器。→ **全部 `gen_*` host 生成器编译链接成功**。
5. **生成器数值 arch**：kernel 生成器 `--archs` 用 `std::stoi` 解析，`ivcore11` 崩溃；解耦为「生成器用 sm80、device 目标 ivcore11」。CoreX CMake 又要求 `CMAKE_CUDA_ARCHITECTURES∈{ivcore10/11/20}` 且各 op 子目标校验不一致、`test/unit` 链接的 `CUDA::nvml` 目标缺失——这些均为构建管线层(workaround-able)，但因下方 terminal 而 moot。

## Failure Gate 结果
按 SOP 先复现墙、再分类、对 workaround-able 实际尝试：
- 逐一解决了 5 类构建层障碍后，对**真实 CUTLASS kernel** `src/comm_none/cutlass_blockscale_gemm_impl.cu` 直接 device 编译（依赖全部补齐）：

  ```
  cute/arch/util.hpp:108:32: error: use of undeclared identifier '__cvta_generic_to_shared'
  ```

  即 CUTLASS CuTe 地基 `cast_smem_ptr_to_uint`（**所有** CUTLASS 3.x kernel 都要用）依赖的 Ampere 共享内存地址内建 `__cvta_generic_to_shared` 在 ivcore11 未声明。
- 裸 ISA 探针（`corex_port/build/probes/`）进一步实证 ivcore11 后端拒绝 Flux 的地基指令：
  - `mma.sync.m16n8k16`（sm80 张量核）→ llc 崩
  - `wgmma.mma_async`（sm90 warp-group mma）→ llc 崩
  - `cp.async.bulk.tensor`（Hopper TMA）→ unknown constraint / llc 崩
  - `cp.async`（Ampere 异步拷贝）→ invalid instruction；`cuda_pipeline.h` 头缺失

**分类：terminal（硬件代差）。** ivcore11 是 Volta 级，不实现 Ampere/Hopper 的张量核与异步拷贝 ISA；Flux 无 Volta 代码路径，整库以这些指令为地基与设计核心。在「禁改 `/usr/local/corex`、禁用 `-O0`/降低正确性」的红线下无 repo-local 规避方案。

## 结论（状态）
- `compile_status = failed`：核心 kernel device 编译被 terminal 硬件代差阻断。
- `test_status = not_attempted`，`tests_run = 0`：全部运行时测试(test/python 分布式用例 + test/unit C++ 单测)都依赖已编译的 `flux_ths_pybind`/`libflux_cuda.so`；该库编不出，`import flux` 直接 `OSError: libflux_cuda.so` 不存在，pytest 收集 0 用例。**未用编译面数字回填测试计数。**
- `overall_status = blocked`。

多卡说明：Flux 分布式用例需 torchrun/多卡；本机 2×BI-V150 满足 ≤2 卡约束，但因原生扩展未编出，从未进入运行阶段，故不涉及跳过计数。
