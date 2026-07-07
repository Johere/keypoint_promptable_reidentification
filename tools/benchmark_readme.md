# KPR Benchmark

两个 benchmark 脚本，测量 KPR 模型的推理效率并同步采集系统资源（CPU / NPU / 内存）：

- `tools/benchmark_kpr_cpu.py` —— PyTorch 原模型，CPU。
- `tools/benchmark_kpr_ov.py` —— OpenVINO IR（`fp16` / `fp16-int8`），CPU / NPU / GPU。

两者共用同一套测量框架（`tools/monitor-system.sh` 系统监控 + 计时循环 + CSV 写出），
每次运行**只产出两份文件**：一份单行汇总 CSV 和一份 `report.log`。

先激活虚拟环境（下文命令默认已激活）：

```bash
source ~/python3-venv/kpr-reid/bin/activate
```

## PyTorch CPU benchmark

在仓库根目录执行：

```bash
python tools/benchmark_kpr_cpu.py
```

默认行为：

- 推理设备：强制 CPU（`CUDA_VISIBLE_DEVICES=""`，模型放到 CPU）。
- 测试时长：`60` 秒。
- 输入样本：`assets/demo/soccer_players/group*/images` 与对应 `keypoints`。
- 配置文件：`configs/kpr/imagenet/kpr_occ_posetrack_test.yaml`。
- 系统监控：每 `5` 秒采样一次。
- 产物：`results/kpr_cpu_benchmark_<timestamp>.csv` + 同名 `.report.log`。

### 快速短测

用于确认脚本能跑通，不作为正式性能结果：

```bash
python tools/benchmark_kpr_cpu.py \
  --duration 10 --limit-samples 1 --warmup-iters 0 --monitor-interval 3 \
  --output-csv results/kpr_cpu_smoke.csv
```

### 常用参数

- `--duration`：正式计时推理时长（秒），默认 `60`。
- `--batch-size`：每次推理的样本数；`0` 表示把全部 demo 样本作为一个 batch。
- `--warmup-iters`：正式计时前的 warmup 轮数，默认 `1`。
- `--monitor-interval`：`tools/monitor-system.sh` 采样间隔（秒），默认 `5`。
- `--torch-threads` / `--torch-interop-threads`：`torch.set_num_threads()` /
  `torch.set_num_interop_threads()`，`0` 表示 PyTorch 默认。
- `--limit-samples`：限制使用的样本数；`0` 表示全部。
- `--output-csv`：汇总 CSV 输出路径；`report.log` 默认取同名 `.report.log`。
- `--log-file`：自定义 `report.log` 路径。
- `--no-monitor`：不启动系统监控（此时 CPU/NPU/内存列为空）。

## OpenVINO benchmark（CPU / NPU / GPU）

```bash
# CPU
python tools/benchmark_kpr_ov.py --device CPU --precision fp16       --duration 60
python tools/benchmark_kpr_ov.py --device CPU --precision fp16-int8  --duration 60

# GPU（核显）
python tools/benchmark_kpr_ov.py --device GPU --precision fp16       --duration 60

# NPU（静态 batch=1；首次编译较慢，用 --cache-dir 加速重复运行）
python tools/benchmark_kpr_ov.py --device NPU --precision fp16-int8  --duration 60 \
  --cache-dir openvino_models/.ov_cache
```

产物：`results/kpr_ov_benchmark_<device>_<precision>_<timestamp>.csv` + 同名 `.report.log`。

### OV 专有参数

在 PyTorch 脚本参数（`--duration`、`--batch-size`、`--warmup-iters`、`--monitor-interval`、
`--limit-samples`、`--output-csv`、`--log-file`、`--no-monitor`）之外，额外提供：

- `--device`：OpenVINO 设备，`CPU` / `NPU` / `GPU` / `AUTO`，默认 `CPU`。
- `--precision`：`fp16` 或 `fp16-int8`，默认 `fp16`（决定默认加载的 IR 路径）。
- `--model`：直接指定 `.xml` 路径，覆盖 `--precision` 的默认路径。
- `--static-batch`：把 IR reshape 成固定 batch 后再编译。`-1`=自动（NPU→batch 大小，
  其余设备→动态）；`0`=动态（仅 CPU/GPU 支持）；`N`=固定为 N。
- `--cache-dir`：OpenVINO `CACHE_DIR`，缓存编译结果，摊薄 NPU 多秒级编译开销。

> **NPU 说明**：NPU 无法编译动态形状的 IR，脚本会自动把模型 reshape 成固定 batch（默认 1）
> 再编译；`--batch-size` 在 NPU 上会被规整为该固定 batch。

## 汇总 CSV 字段

汇总 CSV 是**单行**结果，严格只包含以下列：

| 列 | 含义 |
|---|---|
| `device` | 后端 + 设备 + 精度，如 `pytorch:CPU`、`openvino:CPU:fp16`、`openvino:NPU:fp16-int8` |
| `batch_size` | 每次推理样本数 |
| `model_input` | 输入张量 shape |
| `model_output` | 输出张量 shape |
| `fps` | 每秒处理图片数（`timed_samples / duration_actual`） |
| `latency_ms_avg` | 单次推理平均耗时（ms） |
| `cpu_pct_avg` / `cpu_pct_max` | 整机 CPU 使用率 |
| `npu_pct_avg` / `npu_pct_max` | NPU 使用率（来自 `intel_vpu` sysfs） |
| `mem_used_gib_max` | 峰值已用内存（GiB） |

其余细节（时间戳、config/权重/模型路径、warmup、线程数、输入样本数、latency 的
min/p50/p95/max、平均内存、swap、每次监控采样行等）都写入 `report.log`。

## 查看结果

```bash
cat results/kpr_ov_benchmark_*.csv          # 汇总 CSV
tail -n 80 results/kpr_ov_benchmark_*.report.log   # 完整日志（含监控输出）
```

## 注意事项

- `tools/monitor-system.sh` 的 CPU 使用率来自 `/proc/stat` 聚合 CPU 行，是整机平均，不是单进程。
- NPU 使用率来自 `intel_vpu` 的 `npu_busy_time_us`；只有走 NPU 的运行 `npu_pct_*` 才非零。
- 短测时间过短时监控只有 1~2 行采样，正式对比建议 `--duration 60` 或更长。
- 原始系统监控 CSV 写在临时目录、解析后删除，不作为产物；如需原始采样请看 `report.log`。
