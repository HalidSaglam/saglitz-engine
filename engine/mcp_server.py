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


if __name__ == "__main__":
    mcp.run()
