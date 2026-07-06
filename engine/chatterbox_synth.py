#!/usr/bin/env python3
"""Standalone Chatterbox Multilingual TTS worker — runs INSIDE the isolated
tools-chatterbox venv, invoked as a subprocess by the main engine.

Chatterbox-mlx pulls transformers 4.x + torch 2.8, which conflict with the main
engine's mflux (transformers 5.x). So it lives in its own venv and we talk to it
over argv/JSON only — never imported into engine-venv.

Usage:
  python chatterbox_synth.py --text "..." --lang tr --out out.wav [--ref ref.wav]
                             [--cfg 0.7] [--temp 0.6] [--exaggeration 0.4]
Emits one JSON line on stdout: {"ok": true, "sr": 24000, "dur": 5.6} or
{"ok": false, "error": "..."}.
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None)          # reference audio → voice clone
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--exaggeration", type=float, default=0.4)
    a = ap.parse_args()
    try:
        # resemble-perth 1.0.1 ships PerthImplicitWatermarker = None on some
        # setups; fall back to the no-op DummyWatermarker so init doesn't crash.
        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
        import numpy as np
        import soundfile as sf
        from chatterbox import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_pretrained(device="mps")
        kw = dict(language_id=a.lang, cfg_weight=a.cfg,
                  temperature=a.temp, exaggeration=a.exaggeration)
        if a.ref:
            kw["audio_prompt_path"] = a.ref         # zero-shot clone of this voice
        wav = model.generate(a.text, **kw)
        audio = np.array(wav).squeeze()
        sf.write(a.out, audio, model.sr)
        print(json.dumps({"ok": True, "sr": int(model.sr),
                          "dur": round(len(audio) / model.sr, 2)}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)[:300]}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
