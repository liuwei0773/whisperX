<h1 align="center">WhisperX</h1>

Fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend, requires <8GB gpu memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio) (speaker ID labels)
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation

**Pipeline:** Whisper transcription → wav2vec2 forced phoneme alignment (word-level timestamps) → pyannote VAD + speaker diarization.

## Setup ⚙️

### 0. CUDA (GPU only)

For GPU acceleration, install CUDA toolkit 12.8 first. Skip if using CPU only.
- **Linux**: [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- **Windows**: [CUDA 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-1-download-archive)

### 1. Install

From PyPI:

```bash
pip install whisperx
```

From source (development):

```bash
git clone https://github.com/m-bain/whisperX.git
cd whisperX
uv sync --all-extras --dev
```

You may also need `ffmpeg`, `rust`, etc. See [openAI setup](https://github.com/openai/whisper#setup).

### 2. Speaker Diarization

To enable diarization, pass a Hugging Face access token (read) via `--hf_token` and accept the user agreement for the [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) model.

## Usage 💬 (command line)

```bash
# default params, whisper small; add --highlight_words True to visualise word timings in the .srt
whisperx path/to/audio.wav

# bigger model for higher timestamp accuracy (more gpu mem)
whisperx path/to/audio.wav --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4

# speaker labels (set count if known: --min_speakers 2 --max_speakers 2)
whisperx path/to/audio.wav --model large-v2 --diarize --highlight_words True

# CPU / macOS
whisperx path/to/audio.wav --compute_type int8 --device cpu
```

### Other languages

The phoneme alignment model is language-specific. Pass `--language` and use `--model large`. Defaults are provided for `{en, fr, de, es, it}` via torchaudio and many more via Hugging Face — see `DEFAULT_ALIGN_MODELS_HF` in [alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py). If your language isn't listed, find a phoneme-based ASR model on the [HF model hub](https://huggingface.co/models) and test it.

```bash
whisperx --model large-v2 --language de path/to/audio.wav
```

See more examples in [EXAMPLES.md](EXAMPLES.md).

## Python usage 🐍

```python
import whisperx
from whisperx.diarize import DiarizationPipeline

device = "cuda"
audio_file = "audio.mp3"
batch_size = 16        # reduce if low on GPU mem
compute_type = "float16"  # "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])  # before alignment

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print(result["segments"])  # after alignment

# 3. Assign speaker labels
diarize_model = DiarizationPipeline(token=YOUR_HF_TOKEN, device=device)
diarize_segments = diarize_model(audio)  # or diarize_model(audio, min_speakers=..., max_speakers=...)
result = whisperx.assign_word_speakers(diarize_segments, result)
print(result["segments"])  # now with speaker IDs
```

> To free GPU memory between stages: `import gc, torch; gc.collect(); torch.cuda.empty_cache(); del model`

## Technical Details 👷‍♂️

For batching, alignment, and VAD details see the [paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

To reduce GPU memory (2 & 3 can affect quality):
1. reduce batch size, e.g. `--batch_size 4`
2. smaller ASR model, e.g. `--model base`
3. lighter compute type, e.g. `--compute_type int8`

Differences from openai's whisper:
1. Transcription without timestamps (`--without_timestamps True`) to enable single-pass batching; can cause minor discrepancies vs default whisper output.
2. VAD-based segment transcription instead of buffered transcription — reduces WER and enables batched inference.
3. `--condition_on_prev_text` defaults to `False` (reduces hallucination).

## Limitations ⚠️

- Words without characters in the alignment model's dictionary (e.g. "2014." or "£13.60") cannot be aligned and get no timing.
- Overlapping speech is not handled well.
- Diarization is far from perfect.
- A language-specific wav2vec2 model is required for alignment.

## Citation

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```

Built on [openAI's whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/guillaumekln/faster-whisper) / [CTranslate2](https://github.com/OpenNMT/CTranslate2), and [pyannote-audio](https://github.com/pyannote/pyannote-audio). Original author: Max Bain (maxhbain@gmail.com).
