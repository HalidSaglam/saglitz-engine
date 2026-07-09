"""Saglitz Studio — MCP server (lets Claude Code generate images).

Thin client over stdio that forwards to the running Saglitz engine
(`engine/server.py`, default http://127.0.0.1:8765), which holds the FLUX model
in memory. No model is loaded here, so it adds no extra RAM.

Run (Claude Code spawns this over stdio):
    ./engine-venv/bin/python engine/mcp_server.py
"""

from __future__ import annotations

import os
import time

import httpx
from mcp.server.fastmcp import FastMCP, Image

# Default to the always-on Python engine (8765) so local generation via Claude
# just works. Point at SaglitzServer/saglitz-host (8770) with SAGLITZ_ENGINE_URL
# when you want cloud (BYOK) models through the MCP too.
ENGINE_URL = os.environ.get("SAGLITZ_ENGINE_URL", "http://127.0.0.1:8765").rstrip("/")
# FLUX on MLX can take a couple of minutes per image, so allow a long read.
TIMEOUT = httpx.Timeout(connect=5.0, read=900.0, write=30.0, pool=5.0)

mcp = FastMCP("saglitz-photo-studio")


def _hint(exc: Exception) -> str:
    return (
        f"Saglitz motoruna ulaşılamadı ({ENGINE_URL}): {exc}. "
        "Önce motoru başlat (Saglitz Studio app'i aç ya da engine/server.py'yi çalıştır)."
    )


@mcp.tool()
def saglitz_status() -> str:
    """Saglitz görsel motorunun durumunu döndürür (loading/ready/error), model ve quantize bilgisi."""
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            s = c.get(f"{ENGINE_URL}/api/status").json()
        return f"durum={s.get('status')} detay={s.get('detail') or s.get('model') or '-'}"
    except Exception as exc:
        return _hint(exc)


@mcp.tool()
def generate_image(
    prompt: str,
    project: str | None = None,
    model: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    seed: int | None = None,
    guidance: float = 3.5,
    negative_prompt: str | None = None,
):
    """Yerel olarak (mflux / Apple MLX) bir görsel üretir ve görseli döndürür.

    Args:
        prompt: Görsel açıklaması (düz metin).
        project: Görselin kaydedileceği proje/klasör adı (ör. çağıran uygulamanın
            adı). Görsel projects/<project>/ altına düşer. Verilmezse "Genel".
            Aynı iş için hep aynı adı kullan ki o işin görselleri bir arada olsun.
        model: Hangi model — "schnell" (hızlı FLUX, varsayılan), "dev" (FLUX,
            daha kaliteli/yavaş) ya da "z-image-turbo" (foto-gerçekçilik/portre,
            ~9 adım, negative_prompt destekler). Verilmezse motorun varsayılanı.
        width: Genişlik (256–2048, 16'nın katı). Varsayılan 1024.
        height: Yükseklik (256–2048, 16'nın katı). Varsayılan 1024.
        steps: Difüzyon adımı; verilmezse modelin varsayılanı (schnell ~4,
            z-image-turbo ~9, dev ~20).
        seed: Tekrarlanabilirlik için tam sayı; verilmezse rastgele.
        guidance: Prompt'a bağlılık (yalnız dev için anlamlı; schnell ve
            z-image-turbo distile olduğu için yok sayılır).
        negative_prompt: İstenmeyen öğeler (yalnız z-image-turbo kullanır).

    Returns:
        Üretilen görsel (PNG) + üretim özeti.
    """
    payload = {"prompt": prompt, "project": project, "model": model,
               "width": width, "height": height, "steps": steps, "seed": seed,
               "guidance": guidance, "negative_prompt": negative_prompt}
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = _post_retry_busy(c, f"{ENGINE_URL}/api/generate", payload)
    except Exception as exc:
        return _hint(exc)

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return f"Üretim başarısız ({r.status_code}): {detail}"

    data = r.json()
    proj = data.get("project", "Genel")
    out_path = os.path.join(_projects_root(), proj, data["file"])
    summary = (
        f"Üretildi [{data.get('model', '?')} · proje: {proj}]: "
        f"{data['width']}×{data['height']}, {data.get('steps') or '—'} adım, "
        f"seed {data['seed']}, {data.get('elapsed_sec', 0)}s. Dosya: {out_path}"
    )
    try:
        return [summary, Image(path=out_path)]
    except Exception:
        return summary


def _projects_root() -> str:
    """The engine's CURRENT output root (it moved to Application Support); fall back
    to the legacy Desktop folder only if the engine is unreachable."""
    try:
        with httpx.Client(timeout=5.0) as c:
            return c.get(f"{ENGINE_URL}/api/config/output").json()["path"]
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "projects")


def _result_image(data: dict):
    """Build a [summary, Image] result from an engine generation response."""
    proj = data.get("project", "Genel")
    out_path = os.path.join(_projects_root(), proj, data["file"])
    summary = (f"Tamam [{data.get('model', '?')} · proje: {proj}]: "
               f"{data.get('width')}×{data.get('height')}, "
               f"{data.get('elapsed_sec', 0)}s. Dosya: {out_path}")
    try:
        return [summary, Image(path=out_path)]
    except Exception:
        return summary


def _post_retry_busy(client: httpx.Client, url: str, payload: dict, tries: int = 60, wait: float = 5.0):
    """POST, but when the engine is busy (HTTP 429 — it does ONE generation at a
    time) wait and retry so concurrent apps queue politely instead of failing.
    Up to ~5 min of waiting; any non-429 status returns immediately."""
    r = None
    for _ in range(tries):
        r = client.post(url, json=payload)
        if r.status_code != 429:
            return r
        time.sleep(wait)
    return r


def _post(path: str, payload: dict):
    with httpx.Client(timeout=TIMEOUT) as c:
        return _post_retry_busy(c, f"{ENGINE_URL}{path}", payload)


@mcp.tool()
def list_models() -> str:
    """İndirilmiş yerel modelleri ve düzenleme/edit modellerini listeler (üretim için id'leriyle)."""
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            cfg = c.get(f"{ENGINE_URL}/api/config").json()
        ids = [m["id"] for m in cfg.get("models", [])]
        edit = [i for i in ids if "edit" in i or "kontext" in i or "pix2pix" in i]
        return "Kullanılabilir modeller: " + ", ".join(ids) + (
            f"\nDüzenleme modelleri: {', '.join(edit)}" if edit else "")
    except Exception as exc:
        return _hint(exc)


@mcp.tool()
def describe_image(project: str, file: str) -> str:
    """Bir görseli yerel görü modeliyle (BLIP) analiz edip açıklayıcı bir prompt döndürür.

    Args:
        project: Görselin bulunduğu proje/klasör adı.
        file: Görsel dosya adı (projects/<project>/ altındaki).
    """
    try:
        r = _post("/api/interrogate", {"project": project, "file": file})
    except Exception as exc:
        return _hint(exc)
    if r.status_code != 200:
        return f"Yorumlama başarısız ({r.status_code}): {r.text}"
    return r.json().get("prompt", "")


@mcp.tool()
def upscale_image(project: str, file: str, scale: int = 2):
    """Bir görseli büyütür (2x/4x) ve büyütülmüş görseli döndürür.

    Args:
        project: Proje/klasör adı.
        file: Görsel dosya adı.
        scale: Büyütme katsayısı (2–4). Varsayılan 2.
    """
    try:
        r = _post("/api/upscale", {"project": project, "file": file, "scale": scale})
    except Exception as exc:
        return _hint(exc)
    if r.status_code != 200:
        return f"Büyütme başarısız ({r.status_code}): {r.text}"
    return _result_image(r.json())


@mcp.tool()
def edit_image(project: str, file: str, instruction: str, model: str):
    """Bir görseli talimatla düzenler (ör. "gözleri mavi yap"). İndirilmiş bir edit
    modeli (Kontext / Qwen-Image-Edit / Pix2Pix) gerekir — list_models ile bak.

    Args:
        project: Proje/klasör adı.
        file: Düzenlenecek görsel dosya adı.
        instruction: Düzenleme talimatı (İngilizce önerilir).
        model: Edit modeli id'si (list_models'tan).
    """
    try:
        r = _post("/api/edit", {"project": project, "file": file,
                                "instruction": instruction, "model": model})
    except Exception as exc:
        return _hint(exc)
    if r.status_code != 200:
        return f"Düzenleme başarısız ({r.status_code}): {r.text}"
    return _result_image(r.json())


# --- Helpers for media (audio/video) results (can't embed as Image) ------------
def _media_result(data: dict, kind: str = "dosya") -> str:
    proj = data.get("project", "Genel")
    out_path = os.path.join(_projects_root(), proj, data["file"])
    extra = []
    for k in ("duration", "lang", "bpm", "frames", "removed"):
        if data.get(k) is not None:
            extra.append(f"{k}={data[k]}")
    return f"✓ {kind} hazır [proje: {proj}] {' '.join(extra)}. Dosya: {out_path}"


def _get(path: str):
    with httpx.Client(timeout=TIMEOUT) as c:
        return c.get(f"{ENGINE_URL}{path}")


def _err(r) -> str:
    try:
        return f"Başarısız ({r.status_code}): {r.json().get('detail', r.text)}"
    except Exception:
        return f"Başarısız ({r.status_code}): {r.text}"


# === Audio ======================================================================
@mcp.tool()
def text_to_speech(text: str, project: str | None = None,
                   voice: str = "af_heart", engine: str = "kokoro") -> str:
    """Metni yerel olarak seslendirir (Kokoro/Piper TTS) ve .wav döndürür.

    Args:
        text: Seslendirilecek metin.
        project: Proje/klasör adı (verilmezse "Genel").
        voice: Ses id'si (Kokoro: af_heart, am_adam… — voices için list_voices).
        engine: "kokoro" (varsayılan, çok dilli) | "piper".
    """
    try:
        r = _post("/api/audio/tts", {"text": text, "project": project,
                                     "voice": voice, "engine": engine})
    except Exception as exc:
        return _hint(exc)
    return _media_result(r.json(), "Seslendirme") if r.status_code == 200 else _err(r)


@mcp.tool()
def speak_multilingual(text: str, lang: str = "tr", project: str | None = None,
                       ref_file: str | None = None) -> str:
    """Ticari-güvenli yerel çok-dilli ses (ACE-Step değil, Chatterbox) — Türkçe,
    Almanca + 21 dil, isteğe bağlı ses klonlama. Motorun app'ten bir kez kurulması
    gerekir (~2.6 GB).

    Args:
        text: Seslendirilecek metin (seçilen dilde).
        lang: Dil kodu — tr, de, en, es, fr, it, pt, nl, pl, ru, ar, ja, ko, zh, hi…
        project: Proje/klasör adı.
        ref_file: (İsteğe bağlı) projedeki 5–10 sn'lik bir ses dosyası — o sesi klonlar.
    """
    try:
        if not _get("/api/audio/chatterbox").json().get("installed"):
            return ("Chatterbox kurulu değil. Saglitz app → Ses stüdyosu → 'Türkçe/DE ✨' "
                    "→ Set up (~2.6 GB) ile bir kez kur, sonra tekrar dene.")
        r = _post("/api/audio/chatterbox/tts",
                  {"text": text, "lang": lang, "project": project, "ref_file": ref_file})
    except Exception as exc:
        return _hint(exc)
    return _media_result(r.json(), f"{lang} sesi") if r.status_code == 200 else _err(r)


@mcp.tool()
def generate_music(prompt: str, duration: int = 30, project: str | None = None,
                   lyrics: str = "[Instrumental]") -> str:
    """Yerel, ticari-güvenli müzik üretir (ACE-Step). Motorun app'ten bir kez
    kurulması gerekir (~15 GB).

    Args:
        prompt: Müzik tarifi (ör. "upbeat corporate", "lofi hip hop study beats").
        duration: Süre (saniye, 10–240). Varsayılan 30.
        project: Proje/klasör adı.
        lyrics: Şarkı sözü; enstrümantal için "[Instrumental]" (varsayılan).
    """
    try:
        if not _get("/api/audio/music").json().get("installed"):
            return ("ACE-Step kurulu değil. Saglitz app → Ses stüdyosu → 'Music 🎵' "
                    "→ Set up (~15 GB) ile bir kez kur, sonra tekrar dene.")
        with httpx.Client(timeout=httpx.Timeout(connect=5, read=1800, write=30, pool=5)) as c:
            r = c.post(f"{ENGINE_URL}/api/audio/music/generate",
                       json={"prompt": prompt, "duration": duration,
                             "project": project, "lyrics": lyrics})
    except Exception as exc:
        return _hint(exc)
    return _media_result(r.json(), "Müzik") if r.status_code == 200 else _err(r)


@mcp.tool()
def transcribe_audio(project: str, file: str) -> str:
    """Bir ses dosyasını metne çevirir (yerel Whisper/Parakeet).

    Args:
        project: Proje/klasör adı.
        file: Ses dosya adı (projects/<project>/ altındaki).
    """
    try:
        r = _post("/api/audio/transcribe", {"project": project, "ref_file": file})
    except Exception as exc:
        return _hint(exc)
    return r.json().get("text", "") if r.status_code == 200 else _err(r)


@mcp.tool()
def enhance_audio(project: str, file: str) -> str:
    """Bir ses klibini temizler (gürültü azaltma + de-ess + seviye normalizasyonu).

    Args:
        project: Proje/klasör adı.
        file: Ses dosya adı.
    """
    try:
        r = _post("/api/audio/enhance", {"project": project, "file": file})
    except Exception as exc:
        return _hint(exc)
    return _media_result(r.json(), "Temizlenmiş ses") if r.status_code == 200 else _err(r)


# === Video ======================================================================
@mcp.tool()
def generate_video(prompt: str, project: str | None = None,
                   frames: int = 49, image_file: str | None = None) -> str:
    """Yerel olarak kısa bir video üretir (Wan/LTX). İndirilmiş bir video modeli
    gerekir; en kaliteli indirilmiş model otomatik seçilir. GPU-ağır: dakikalar sürer.

    Args:
        prompt: Klip tarifi (ör. "a red lighthouse at sunset, waves, cinematic").
        project: Proje/klasör adı.
        frames: Kare sayısı (9–121; ~16 fps, yani 49 kare ≈ 3 sn). Varsayılan 49.
        image_file: (İsteğe bağlı) projedeki bir görsel — image-to-video başlangıç karesi.
    """
    try:
        vids = [m for m in _get("/api/dt/models").json()
                if m.get("downloaded") and any(k in m["ckpt"].lower()
                                               for k in ("wan", "ltx", "ti2v"))]
        if not vids:
            return "İndirilmiş video modeli yok. Saglitz app → Models'tan bir Wan/LTX modeli indir."
        vids.sort(key=lambda m: (("5b" in m["ckpt"] or "ltx" in m["ckpt"]), not ("1.3b" in m["ckpt"])), reverse=True)
        model = vids[0]["ckpt"]
        payload = {"model": model, "prompt": prompt, "project": project,
                   "frames": frames, "smooth": True}
        if image_file:
            payload["image_path"] = os.path.join(_projects_root(), project or "Genel", image_file)
        with httpx.Client(timeout=httpx.Timeout(connect=5, read=1800, write=30, pool=5)) as c:
            r = _post_retry_busy(c, f"{ENGINE_URL}/api/generate/video", payload)
    except Exception as exc:
        return _hint(exc)
    return _media_result(r.json(), "Video") if r.status_code == 200 else _err(r)


@mcp.tool()
def talking_head(project: str, image_file: str, audio_file: str) -> str:
    """Bir portre + bir konuşma klibinden dudak-senkronlu konuşan video üretir
    (fal.ai bulut — app'te bir fal.ai anahtarı gerekir).

    Args:
        project: Proje/klasör adı (görsel ve ses bu projede olmalı).
        image_file: Portre görsel dosya adı.
        audio_file: Konuşma sesi dosya adı (ör. text_to_speech / speak_multilingual çıktısı).
    """
    try:
        r = _post("/api/video/talkinghead",
                  {"project": project, "image_file": image_file, "audio_file": audio_file})
    except Exception as exc:
        return _hint(exc)
    return _media_result(r.json(), "Konuşan video") if r.status_code == 200 else _err(r)


# === Image (extra) ==============================================================
@mcp.tool()
def remove_background(project: str, file: str):
    """Bir görselin arka planını kaldırır (rembg/BiRefNet) ve şeffaf PNG döndürür.

    Args:
        project: Proje/klasör adı.
        file: Görsel dosya adı.
    """
    try:
        r = _post("/api/image/removebg", {"project": project, "file": file})
    except Exception as exc:
        return _hint(exc)
    return _result_image(r.json()) if r.status_code == 200 else _err(r)


# === Projects ===================================================================
@mcp.tool()
def list_projects() -> str:
    """Mevcut projeleri (klasörleri) listeler."""
    try:
        cfg = _get("/api/config").json()
    except Exception as exc:
        return _hint(exc)
    projs = cfg.get("projects", [])
    return "Projeler: " + (", ".join(projs) if projs else "(yok)")


if __name__ == "__main__":
    mcp.run()
