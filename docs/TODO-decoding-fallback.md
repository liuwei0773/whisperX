# TODO: 为批量推理补回解码兜底（温度回退 / 质量判定）

状态：设计阶段，待决策后实现
相关代码：`whisperx/asr.py`、`whisperx/transcribe.py`、`whisperx/__main__.py`

## 背景

WhisperX 为了批量并行加速，用自写的 `generate_segment_batched`
(`asr.py:37-93`) 直接调底层 `self.model.generate`，**绕开了 faster-whisper
的 `generate_with_fallback`**。

代价（均已从代码确认）：

1. 模型幻觉 / 复读片段会原样进入最终结果，无任何自动补救。
2. `avg_logprob` 算了但没人用（`asr.py:78-82`），低质量片段无标记。
3. 一批 CLI 参数实际是 no-op（只有未被调用的 `generate_with_fallback`
   才会读它们）：
   - `--temperature` / `--temperature_increment_on_fallback`
   - `--compression_ratio_threshold` / `--logprob_threshold` / `--no_speech_threshold`
   - `--best_of`（压根没进 `asr_options`，见 `transcribe.py:99-112`）
   - `--fp16`（精度实际由 `--compute_type` 决定）
4. `condition_on_previous_text` 在 `transcribe.py:107` 被硬编码为 `False`。
5. 段级时间戳来自 VAD（`asr.py:284-285`），非模型（`without_timestamps=True`）。

## 目标

把 faster-whisper 的解码兜底逻辑以**两阶段、对批量友好**的方式补回来，
让上面那批死参数真正生效，同时尽量不牺牲批量速度。

## 参考：faster-whisper v1.2.0 的判定逻辑

`generate_with_fallback` 中的 `needs_fallback`：

```python
needs_fallback = False
if compression_ratio > options.compression_ratio_threshold:   # 默认 2.4
    needs_fallback = True   # 文本过于重复（复读）
if avg_logprob < options.log_prob_threshold:                  # 默认 -1.0
    needs_fallback = True   # 模型置信度过低
if (no_speech_prob > options.no_speech_threshold              # 默认 0.6
        and avg_logprob < options.log_prob_threshold):
    needs_fallback = False  # 判为静音，强制接受

# 循环：for t in temperatures: ... if not needs_fallback: break
# 全失败兜底：max(below_cr_threshold_results or all_results, key=avg_logprob)
```

## 可行性（已确认）

- `no_speech_prob` 可在同一次 `generate` 调用里通过 `return_no_speech_prob=True`
  拿到，不需额外的模型调用。
- `compression_ratio` 用一行 `zlib.compress` 计算：
  ```python
  def get_compression_ratio(text: str) -> float:
      b = text.encode("utf-8")
      return len(b) / len(zlib.compress(b))
  ```
- `avg_logprob` 现已在 `asr.py:78-82` 计算。

## 拟定方案：两阶段

### 阶段 1 — 主循环（保持并行，几乎不改解码次数）

在 `generate_segment_batched` (`asr.py:37-93`)：
1. `self.model.generate(..., return_no_speech_prob=True)`，读 `no_speech_prob`。
2. 用 `zlib` 算每段 `compression_ratio`。
3. 返回值带上 `no_speech_prob` 和 `compression_ratio`（目前只返回 text/avg_logprob）。

在 `transcribe` 循环 (`asr.py:267-288`)：
- 对每段套用 `needs_fallback` 判定。
- 失败片段记下索引，结果先占位；正常片段照常写入。

### 阶段 2 — 重试失败片段

主循环跑完后，对记下的失败片段用更高温度重新解码，
收敛或温度用尽后用兜底逻辑（选 avg_logprob 最高）替换占位结果。

## 性能预估（取决于失败率 p）

| 方案 | 额外开销 | 说明 |
|---|---|---|
| C：只算指标不重试 | ~0% | 仅暴露质量信号，不修复 |
| B：失败片段单独重试 | +5~15%（典型） | 干净音频 p 低；嘈杂可达 +30~100% |
| A：整批统一升温 | 3~6 倍 | 一个拖累全部，不可取 |

建议先做方案 C（近零成本），在真实数据上量出 p，再决定要不要加阶段 2。

## 待决策（实现前需拍板）

1. **重试解码方式**：当前批量路径用 beam search（beam_size=5）。重试时：
   - [ ] 高温采样（贴近原版 Whisper，但与主循环行为不一致）
   - [ ] 保持 beam search 仅叠加温度（简单一致，但温度对 beam 影响弱）
   - [ ] 低温档 beam、高温档（>0.4）转采样（兼顾，但最复杂）
2. **重试批量策略**：
   - [ ] 失败片段重新组 batch 并行（保留批量优势，但 batch 内仍"一个拖累全部"）
   - [ ] 逐片段独立串行（完全复刻原版语义，失败片段间无并行）
3. **启用方式**：
   - [ ] 新增 CLI 开关，默认关闭（对现有用户零影响，**推荐**）
   - [ ] 默认开启（让现有阈值参数真正生效，但改变现有用户的速度/输出）
4. **范围**：是否先只做方案 C（暴露 compression_ratio / avg_logprob /
   no_speech_prob 到 segment），观察真实失败率后再决定阶段 2？

## 验证清单（实现后）

- [ ] 跑构建 / 现有测试，确认无回归
- [ ] 用一段含已知复读/幻觉的音频，确认失败片段被识别并修复
- [ ] 用一段口语化重复（"啊啊啊"）音频，确认未被误删（靠高 logprob 兜底保留）
- [ ] 测量开启前后的端到端耗时，记录真实 p 与开销
- [ ] 确认死参数（temperature、各阈值）现在确实生效

