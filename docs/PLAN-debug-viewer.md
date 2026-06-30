# 计划：WhisperX 阶段调试可视化界面

## 目标
一个零依赖的本地可视化界面，实时对照播放**视频/音频**、显示**字幕**、并叠加
**各阶段中间产物**（VAD 块 / ASR 段 / 对齐词 / 说话人），发现字幕异常时可直接
在时间轴上点击定位到对应阶段的产物，排查问题出在哪一环。

数据来源 = 上一步已落盘的 `--debug_dir` 四个 JSON：
`*.01_vad_chunks.json` / `*.02_asr.json` / `*.03_align.json` / `*.04_diarize.json`。

## 技术决策（已与用户确认）
- 技术栈：**Python 标准库 `http.server`**（带 Range 支持，满足 `<video>` seek）
  + 单文件 HTML/JS 前端。**不新增 pyproject 依赖**。
- 媒体：`<video>` 播放原始影像 + 下方 canvas 波形/时间轴，叠加各阶段区段。
- 启动：**独立子命令** `python -m whisperx.viewer`，与转写主流程解耦。

## 新增文件
1. `whisperx/viewer/__init__.py` — 导出入口。
2. `whisperx/viewer/__main__.py` — argparse CLI：
   - `--debug_dir`（必填）：读取阶段 JSON 的目录
   - `--media`（必填）：原始视频/音频文件路径
   - `--host`（默认 `127.0.0.1`）、`--port`（默认 `0`=自动选空闲端口）
   - `--no-browser`：不自动打开浏览器
   - 启动后打印 URL，`webbrowser.open` 自动打开。
3. `whisperx/viewer/server.py` — 基于 `http.server.ThreadingHTTPServer` +
   自定义 `BaseHTTPRequestHandler`：
   - `GET /` → 返回前端 HTML（内联，避免额外静态文件路径问题）
   - `GET /api/stages` → 扫描 debug_dir，把存在的阶段 JSON 合并成一个
     `{vad, asr, align, diarize, meta}` 返回（缺某阶段则该键为 null）
   - `GET /media` → **流式 + Range** 返回媒体文件（视频 seek 必需）
   - 安全：handler 只允许这几个固定路由 + 白名单的两个具体文件路径
     （debug_dir 下的 JSON、--media 指定的单个文件），不做任意路径映射，
     防目录穿越。绑定 127.0.0.1。
4. `whisperx/viewer/static/index.html`（或内联进 server.py，<150 行用内联）
   — 前端，纯原生 JS，无构建步骤：
   - 顶部 `<video controls>`（媒体 = `/media`）
   - 中部多轨时间轴（canvas 或绝对定位 div）：
     轨1 VAD 块、轨2 ASR 段（按 avg_logprob 着色，低置信度标红）、
     轨3 对齐词、轨4 说话人。鼠标悬停显示文本/分数；点击 seek 视频到该区段。
   - 底部当前字幕高亮（随 `timeupdate` 滚动）。
   - 一条游标随播放进度在所有轨上移动，便于上下对照同一时刻各阶段产物。

## 关键实现点
- **avg_logprob 着色**：`02_asr.json` 已含 `avg_logprob`，把低于阈值（默认 -1.0）
  的 ASR 段标红——直接对应字幕异常排查（哪段解码置信度低）。
- **时间对齐**：四阶段时间戳同轴（秒），游标统一驱动，便于发现
  「VAD 切块没切对 → ASR 段跟着错 → 字幕异常」这类跨阶段传播问题。
- **波形**：用 WebAudio `decodeAudioData` 在前端从 `/media` 解码画波形，
  纯前端、零后端依赖（视频文件也能解出音轨）。若解码失败则降级为纯区段轨。

## 不做（范围控制）
- 不修改任何现有转写/对齐/说话人逻辑。
- 不引入前端构建工具、不引入 web 框架。
- 不做编辑/回写字幕功能（仅只读排查）。本期只读可视化。

## 验证
- 造一组样例 JSON（mock 四阶段）+ 一个小媒体文件，启动 server：
  - `curl /api/stages` 返回合并 JSON
  - `curl -H 'Range: bytes=0-99' /media` 返回 206 + 正确分片（证明 seek 可用）
  - 路径穿越请求（如 `/media?../../etc/passwd` 变体）被拒
- 前端在浏览器手动确认：视频播放、波形渲染、四轨叠加、点击 seek、字幕高亮。
- 因当前环境无 torch，无法跑真实转写产出 JSON；用 mock JSON 验证 viewer，
  并说明这一限制。

## 风险
- 不同浏览器对 `file://` 限制无关（我们走 http）。
- 大视频 WebAudio 解码可能慢/占内存 → 降级策略已含（区段轨仍可用）。
