"""The single-page frontend for the debug viewer, as an HTML string.

Kept as a Python module (rather than a static file) so the zero-dependency
server has nothing to resolve on disk. Pure vanilla JS, no build step.
"""

# NOTE: built up in parts to stay within edit-size limits; concatenated at end.
_HTML_HEAD = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WhisperX 阶段调试</title>
<style>
  :root { --bg:#11141a; --panel:#1b1f29; --fg:#e6e9ef; --muted:#8b94a7;
          --line:#2a3040; --accent:#4da3ff; --bad:#ff5d5d; --ok:#46c46e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }
  header { padding:8px 14px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:14px; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .meta { color:var(--muted); font-size:12px; }
  #wrap { display:grid; grid-template-columns: 1fr 360px; gap:12px; padding:12px; }
  video { width:100%; background:#000; border-radius:6px; max-height:42vh; }
  #tracks { margin-top:10px; background:var(--panel); border:1px solid var(--line);
            border-radius:6px; padding:8px; position:relative; overflow-x:auto; }
  .track { position:relative; height:34px; margin:6px 0; background:#151922;
           border-radius:4px; }
  .track-label { position:absolute; left:4px; top:3px; font-size:11px;
                 color:var(--muted); z-index:3; pointer-events:none; }
  .seg { position:absolute; top:15px; height:16px; border-radius:3px;
         background:var(--accent); opacity:.8; cursor:pointer; overflow:hidden;
         font-size:10px; white-space:nowrap; color:#06121f; padding:0 3px; }
  .seg:hover { opacity:1; outline:1px solid #fff; }
  .seg.bad { background:var(--bad); }
  #cursor { position:absolute; top:0; bottom:0; width:2px; background:#fff;
            z-index:5; pointer-events:none; }
  #wave { width:100%; height:60px; display:block; background:#0d1016;
          border-radius:4px; margin:6px 0; }
  #side { background:var(--panel); border:1px solid var(--line); border-radius:6px;
          padding:10px; height:78vh; overflow:auto; }
  .sub { padding:6px 8px; border-radius:4px; cursor:pointer; border:1px solid transparent; }
  .sub:hover { background:#222836; }
  .sub.active { background:#27405e; border-color:var(--accent); }
  .sub .t { color:var(--muted); font-size:11px; }
  .sub .lp { float:right; font-size:11px; }
  .sub .lp.bad { color:var(--bad); }
  .legend { font-size:11px; color:var(--muted); margin:4px 0 10px; }
  .pill { display:inline-block; padding:1px 6px; border-radius:8px; margin-right:6px; }
</style>
</head>
<body>
<header>
  <h1>WhisperX 阶段调试</h1>
  <span class="meta" id="meta">加载中…</span>
</header>
<div id="wrap">
  <div id="left">
    <video id="video" controls src="/media"></video>
    <canvas id="wave"></canvas>
    <div id="tracks"></div>
  </div>
  <div id="side">
    <div class="legend">
      <span class="pill" style="background:#4da3ff;color:#06121f">正常段</span>
      <span class="pill" style="background:#ff5d5d;color:#06121f">低置信度(avg_logprob)</span>
      点击任意段/字幕可跳转视频
    </div>
    <div id="subs"></div>
  </div>
</div>
"""

_HTML_SCRIPT = """
<script>
const video = document.getElementById('video');
const tracksEl = document.getElementById('tracks');
const subsEl = document.getElementById('subs');
const metaEl = document.getElementById('meta');
const LP_THRESHOLD = -1.0;   // mirror faster-whisper log_prob_threshold default
let duration = 0;            // timeline span in seconds
let asrSegs = [];            // for subtitle list + highlight

function fmt(t) {
  if (t == null || isNaN(t)) return '?';
  const m = Math.floor(t / 60), s = (t % 60).toFixed(2).padStart(5, '0');
  return m + ':' + s;
}
function makeTrack(label) {
  const t = document.createElement('div');
  t.className = 'track';
  const l = document.createElement('div');
  l.className = 'track-label';
  l.textContent = label;
  t.appendChild(l);
  tracksEl.appendChild(t);
  return t;
}
function addSeg(track, start, end, text, bad) {
  if (duration <= 0) return;
  const seg = document.createElement('div');
  seg.className = 'seg' + (bad ? ' bad' : '');
  seg.style.left = (100 * start / duration) + '%';
  seg.style.width = Math.max(0.2, 100 * (end - start) / duration) + '%';
  seg.title = text ? `[${fmt(start)}–${fmt(end)}] ${text}` : `[${fmt(start)}–${fmt(end)}]`;
  if (text) seg.textContent = text;
  seg.onclick = () => { video.currentTime = start; video.play(); };
  track.appendChild(seg);
}

// ---- render stages ---------------------------------------------------------
function render(data) {
  const found = (data.meta && data.meta.found) || {};
  metaEl.textContent = '阶段: ' + (Object.keys(found).join(', ') || '无') +
                       ' · 目录 ' + (data.meta ? data.meta.debug_dir : '?');

  // Determine timeline span from the widest stage available.
  const ends = [];
  if (data.vad && data.vad.chunks) for (const c of data.vad.chunks) ends.push(c.end);
  if (data.asr && data.asr.segments) for (const s of data.asr.segments) ends.push(s.end);
  if (data.align && data.align.segments) for (const s of data.align.segments) ends.push(s.end);
  duration = Math.max(0, ...ends);
  if (video.duration && isFinite(video.duration)) duration = Math.max(duration, video.duration);

  // Track 1: VAD merged chunks (+ their raw sub-segments as nested ticks)
  if (data.vad && data.vad.chunks) {
    const tr = makeTrack('VAD 块');
    for (const c of data.vad.chunks) addSeg(tr, c.start, c.end, '', false);
  }
  // Track 2: ASR segments, red when avg_logprob below threshold
  if (data.asr && data.asr.segments) {
    asrSegs = data.asr.segments;
    const tr = makeTrack('ASR 段');
    for (const s of data.asr.segments) {
      const bad = (typeof s.avg_logprob === 'number') && s.avg_logprob < LP_THRESHOLD;
      addSeg(tr, s.start, s.end, (s.text || '').trim(), bad);
    }
  }
  // Track 3: aligned sentences (segment-level, not per-word — per-word is too fine)
  if (data.align && data.align.segments) {
    const tr = makeTrack('对齐句');
    for (const s of data.align.segments) {
      const words = s.words || [];
      // Segment may lack start/end; fall back to its first/last word's timing.
      let st = s.start, en = s.end;
      if (typeof st !== 'number' && words.length) st = words[0].start;
      if (typeof en !== 'number' && words.length) en = words[words.length - 1].end;
      if (typeof st === 'number' && typeof en === 'number')
        addSeg(tr, st, en, (s.text || '').trim(), false);
    }
  }
  // Track 4: speakers (from diarized segments)
  if (data.diarize && data.diarize.segments) {
    const tr = makeTrack('说话人');
    for (const s of data.diarize.segments)
      if (s.speaker) addSeg(tr, s.start, s.end, s.speaker, false);
  }

  // Moving cursor across all tracks.
  const cursor = document.createElement('div');
  cursor.id = 'cursor';
  tracksEl.appendChild(cursor);
  video.addEventListener('timeupdate', () => {
    if (duration > 0) cursor.style.left = (100 * video.currentTime / duration) + '%';
    highlightSub();
  });

  renderSubs();
}

// ---- subtitle list ---------------------------------------------------------
let subEls = [];
function renderSubs() {
  subsEl.innerHTML = '';
  subEls = asrSegs.map((s) => {
    const div = document.createElement('div');
    div.className = 'sub';
    const bad = (typeof s.avg_logprob === 'number') && s.avg_logprob < LP_THRESHOLD;
    const lp = (typeof s.avg_logprob === 'number')
      ? `<span class="lp ${bad ? 'bad' : ''}">lp ${s.avg_logprob.toFixed(2)}</span>` : '';
    div.innerHTML = `${lp}<span class="t">[${fmt(s.start)}–${fmt(s.end)}]</span><br>` +
                    (s.text || '').trim();
    div.onclick = () => { video.currentTime = s.start; video.play(); };
    subsEl.appendChild(div);
    return div;
  });
}
function highlightSub() {
  const t = video.currentTime;
  for (let i = 0; i < asrSegs.length; i++) {
    const on = t >= asrSegs[i].start && t < asrSegs[i].end;
    subEls[i].classList.toggle('active', on);
    if (on) subEls[i].scrollIntoView({ block: 'nearest' });
  }
}

// ---- waveform (best-effort, decoded client-side from /media) ---------------
async function drawWave() {
  const cv = document.getElementById('wave');
  const ctx = cv.getContext('2d');
  cv.width = cv.clientWidth; cv.height = 60;
  try {
    const buf = await (await fetch('/media')).arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const audio = await ac.decodeAudioData(buf);
    const ch = audio.getChannelData(0);
    const step = Math.max(1, Math.floor(ch.length / cv.width));
    ctx.fillStyle = '#4da3ff';
    for (let x = 0; x < cv.width; x++) {
      let min = 1, max = -1;
      for (let i = 0; i < step; i++) {
        const v = ch[x * step + i] || 0;
        if (v < min) min = v; if (v > max) max = v;
      }
      const y1 = (1 + min) * cv.height / 2, y2 = (1 + max) * cv.height / 2;
      ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
  } catch (e) {
    ctx.fillStyle = '#8b94a7';
    ctx.fillText('波形不可用（媒体无法解码）— 区段轨仍可用', 8, 32);
  }
}

// ---- boot ------------------------------------------------------------------
async function boot() {
  const data = await (await fetch('/api/stages')).json();
  // Wait for video metadata so duration is accurate, but don't block forever.
  if (video.readyState >= 1) render(data);
  else {
    video.addEventListener('loadedmetadata', () => render(data), { once: true });
    setTimeout(() => { if (!duration) render(data); }, 1500);
  }
  drawWave();
}
boot();
</script>
"""

# Concatenate the parts into the served document.
HTML = (_HTML_HEAD + _HTML_SCRIPT + "\n</body>\n</html>\n").encode("utf-8")
