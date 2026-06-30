import argparse
import importlib.metadata
import platform

import torch

from whisperx.utils import (LANGUAGES, TO_LANGUAGE_CODE, optional_float,
                            optional_int, str2bool)
from whisperx.log_utils import setup_logging


def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Input ──────────────────────────────────────────────────────────────
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")

    # ── Model & device ─────────────────────────────────────────────────────
    model_args = parser.add_argument_group("Model & device")
    model_args.add_argument("--model", default="small", help="name of the Whisper model to use")
    model_args.add_argument("--model_cache_only", type=str2bool, default=False, help="If True, will not attempt to download models, instead using cached models from --model_dir")
    model_args.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    model_args.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device type to use for PyTorch inference (e.g. cpu, cuda)")
    model_args.add_argument("--device_index", default=0, type=int, help="device index to use for FasterWhisper inference")
    model_args.add_argument("--batch_size", default=8, type=int, help="the preferred batch size for inference")
    model_args.add_argument("--compute_type", default="default", type=str, choices=["default", "float16", "float32", "int8"], help="compute type for computation; 'default' uses float16 on GPU, float32 on CPU")
    model_args.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    model_args.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")

    # ── Output ─────────────────────────────────────────────────────────────
    output_args = parser.add_argument_group("Output")
    output_args.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    output_args.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    output_args.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    output_args.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    output_args.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    output_args.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    # ── Transcription ──────────────────────────────────────────────────────
    transcribe_args = parser.add_argument_group("Transcription")
    transcribe_args.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    transcribe_args.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    transcribe_args.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    transcribe_args.add_argument("--hotwords", type=str, default=None, help="hotwords/hint phrases to the model (e.g. \"WhisperX, PyAnnote, GPU\"); improves recognition of rare/technical terms")
    transcribe_args.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

    # ── Decoding ───────────────────────────────────────────────────────────
    decode_args = parser.add_argument_group("Decoding")
    decode_args.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    decode_args.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    decode_args.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    decode_args.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    decode_args.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
    decode_args.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    decode_args.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")
    decode_args.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    # ── Decoding fallback ──────────────────────────────────────────────────
    fallback_args = parser.add_argument_group("Decoding fallback")
    fallback_args.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    fallback_args.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    fallback_args.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    fallback_args.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    # ── Voice activity detection (VAD) ─────────────────────────────────────
    vad_args = parser.add_argument_group("Voice activity detection (VAD)")
    vad_args.add_argument("--vad_method", type=str, default="pyannote", choices=["pyannote", "silero"], help="VAD method to be used")
    vad_args.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    vad_args.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    vad_args.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # ── Alignment ──────────────────────────────────────────────────────────
    align_args = parser.add_argument_group("Alignment")
    align_args.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    align_args.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    align_args.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    align_args.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # ── Diarization ────────────────────────────────────────────────────────
    diarize_args = parser.add_argument_group("Diarization")
    diarize_args.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    diarize_args.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    diarize_args.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    diarize_args.add_argument("--diarize_model", default="pyannote/speaker-diarization-community-1", type=str, help="Name of the speaker diarization model to use")
    diarize_args.add_argument("--speaker_embeddings", action="store_true", help="Include speaker embeddings in JSON output (only works with --diarize)")

    # ── Logging & info ─────────────────────────────────────────────────────
    log_args = parser.add_argument_group("Logging & info")
    log_args.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")
    log_args.add_argument("--log-level", type=str, default=None, choices=["debug", "info", "warning", "error", "critical"], help="logging level (overrides --verbose if set)")
    log_args.add_argument("--debug_dir", type=str, default=None, help="if set, dump each pipeline stage (VAD chunks, ASR, alignment, diarization) as numbered JSON files into this directory for step-by-step debugging")
    log_args.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    log_args.add_argument("--version", "-V", action="version", version=f"%(prog)s {importlib.metadata.version('whisperx')}",help="Show whisperx version information and exit")
    log_args.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})",help="Show python version information and exit")
    # fmt: on

    args = parser.parse_args().__dict__

    log_level = args.get("log_level")
    verbose = args.get("verbose")

    if log_level is not None:
        setup_logging(level=log_level)
    elif verbose:
        setup_logging(level="info")
    else:
        setup_logging(level="warning")

    from whisperx.transcribe import transcribe_task

    transcribe_task(args, parser)


if __name__ == "__main__":
    cli()
