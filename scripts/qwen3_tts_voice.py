#!/usr/bin/env python3
"""Export a cloned voice for Qwen3-TTS as an any-tts VoiceEmbedding JSON.

any-tts does not yet port the Base model's speaker encoder (ECAPA-TDNN) or
the speech tokenizer's encoder, so a voice is computed once here with
mlx-audio and reused by any-tts for every sentence:

    python scripts/qwen3_tts_voice.py <Qwen3-TTS-Base-dir> <reference.wav> \
        "<what the reference says>" <voice.json> [--start S] [--end S]

The JSON holds the speaker embedding (the x-vector, `data`, shape [2048])
and, for in-context cloning, the reference's codec codes (frames x 16) with
its transcript (`prompt`). Without a transcript (pass ""), only the
x-vector is saved; any-tts then clones the timbre without the reference
prompt, which is faster but less faithful.
"""

import argparse
import json

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="Qwen3-TTS Base model directory")
    parser.add_argument("audio", help="reference speech (any rate; resampled to 24 kHz)")
    parser.add_argument("text", help='transcript of the reference, or "" for x-vector only')
    parser.add_argument("output", help="VoiceEmbedding JSON to write")
    parser.add_argument("--start", type=float, default=0.0, help="clip start, seconds")
    parser.add_argument("--end", type=float, default=None, help="clip end, seconds")
    args = parser.parse_args()

    model = load_model(args.model)
    if model.speaker_encoder is None:
        raise SystemExit("this model has no speaker encoder: use a Qwen3-TTS Base model")

    rate = model.sample_rate
    audio = np.array(load_audio(args.audio, sample_rate=rate), dtype=np.float32)
    end = len(audio) if args.end is None else int(args.end * rate)
    audio = mx.array(audio[int(args.start * rate) : end])

    speaker = model.extract_speaker_embedding(audio, sr=rate)
    speaker = np.array(speaker.astype(mx.float32)).reshape(-1)
    voice = {
        "data": [float(v) for v in speaker],
        "shape": [int(speaker.size)],
        "model_type": "qwen3-tts",
    }

    text = args.text.strip()
    if text:
        codes = model.speech_tokenizer.encode(audio[None, None, :])  # [1, 16, frames]
        codes = np.array(codes[0]).T.astype(np.int64)  # [frames, 16]
        voice["prompt"] = {"codes": codes.tolist(), "text": text}
        frames = f", {codes.shape[0]} reference frames"
    else:
        frames = ""

    with open(args.output, "w") as out:
        json.dump(voice, out)
    print(f"{args.output}: {audio.size / rate:.2f} s of audio{frames}")


if __name__ == "__main__":
    main()
