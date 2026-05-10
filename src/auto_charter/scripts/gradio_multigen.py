"""Auto-Charter Gradio App — generación multi-instrumento / multi-dificultad.

Pipeline completo:
  1. Subida de audio (.ogg / .mp3 / .wav)
  2. Metadatos de la canción
  3. Selección de instrumentos + dificultades
  4. Separación de stems por Demucs (una vez por instrumento)
  5. Beat estimation + MERT + Log-Mel por stem (cacheado por instrumento)
  6. Inferencia autoregresiva — nucleus sampling (T=0.95, top-p=0.92)
     El encoder se corre una sola vez por instrumento; el decoder se corre
     para cada combinación (instrumento, dificultad).
  7. Tokens → ChartData → notes.chart multi-track
  8. Audio → song.ogg (ffmpeg si no es .ogg)
  9. song.ini con los diff_* correctos
 10. ZIP descargable listo para Clone Hero

Lanzar:
    python src/auto_charter/scripts/gradio_multigen.py \\
        --checkpoint checkpoints/run1/best [--port 7860]
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
import urllib.request
from pathlib import Path

import click

# ── Mappings ──────────────────────────────────────────────────────────────────

_INSTR_LABEL_TO_KEY = {"Guitar": "guitar", "Bass": "bass", "Drums": "drums"}
_INSTR_KEY_TO_ID    = {"guitar": 0, "bass": 1, "drums": 2}
_DIFF_LABEL_TO_ID   = {
    "Easy": 0, "Medium": 1, "Hard": 2, "Expert": 3,
    "Expert+": 4, "Expert++": 5, "Expert+++": 6,
}
_DIFF_ID_TO_LABEL   = {v: k for k, v in _DIFF_LABEL_TO_ID.items()}
_DIFF_PRIORITY      = {3: 0, 4: 0, 5: 0, 6: 0, 2: 1, 1: 2, 0: 3}


# ── Metadata helpers ────────────────────────────────────────────────────────────

def fetch_metadata_itunes(artist: str, song: str) -> dict | None:
    """Fetch album metadata from iTunes Search API. Returns dict or None."""
    import requests
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{song} {artist}", "entity": "song", "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        t = results[0]
        artwork = t.get("artworkUrl100", "").replace("100x100", "600x600")
        return {
            "album":   t.get("collectionName", ""),
            "year":    t.get("releaseDate", "")[:4],
            "genre":   t.get("primaryGenreName", ""),
            "artwork": artwork,
        }
    except Exception:
        return None


def download_album_art(url: str, dest: Path) -> bool:
    """Download album artwork to dest. Returns True on success."""
    if not url:
        return False
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception:
        return False


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    audio_path: Path,
    song_name: str,
    artist: str,
    album: str,
    genre: str,
    year: str,
    instrument_labels: list[str],
    difficulty_labels: list[str],
    model,
    device: str,
    temperature: float = 0.95,
    top_p: float = 0.92,
    max_new_tokens: int = 8192,
    use_demucs: bool = True,
    log=print,
    cancel_event=None,
    artwork_path: Path | None = None,
) -> tuple[str, Path | None]:
    """Full generation pipeline. Returns (status, zip_path | None)."""

    import numpy as np
    import torch

    from auto_charter.audio.beat_estimator import BeatEstimator
    from auto_charter.audio.logmel import LogMelExtractor
    from auto_charter.audio.mert_extractor import MERTExtractor
    from auto_charter.audio.separator import StemSeparator
    from auto_charter.parsers.chart_parser import ChartData
    from auto_charter.parsers.chart_renderer import render_chart
    from auto_charter.parsers.sync_track import BPMMap, BPMEvent
    from auto_charter.tokenizer.decoder import decode_tokens

    if not audio_path or not audio_path.exists():
        return "Error: no se proporcionó archivo de audio.", None

    year_int = int(year.strip()) if year.strip().isdigit() else 0

    # Build target list: [(instr_key, diff_id)]
    targets: list[tuple[str, int]] = []
    for instr_label in instrument_labels:
        instr_key = _INSTR_LABEL_TO_KEY.get(instr_label)
        if instr_key is None:
            continue
        for diff_label in difficulty_labels:
            diff_id = _DIFF_LABEL_TO_ID.get(diff_label)
            if diff_id is None:
                continue
            targets.append((instr_key, diff_id))

    if not targets:
        return "Error: selecciona al menos un instrumento y una dificultad.", None

    needed_instrs = list({instr for instr, _ in targets})
    tmpdir = Path(tempfile.mkdtemp(prefix="autocharter_"))

    # Persist ZIP in a stable location so Gradio can serve it
    output_dir = Path("./outputs").resolve()
    output_dir.mkdir(exist_ok=True)

    try:
        # ── 1. Stem separation (opcional) ─────────────────────────────────────
        if use_demucs:
            log("🎸 Separando stems con Demucs ...")
            stems_dir = tmpdir / "stems"
            try:
                stem_paths = StemSeparator(device=device).separate(
                    audio_path, stems_dir, instruments=needed_instrs
                )
            except Exception as e:
                log(f"  ⚠ Demucs falló ({e}). Usando audio original.")
                stem_paths = {i: audio_path for i in needed_instrs}
        else:
            log("🎸 Usando audio completo (sin Demucs) ...")
            stem_paths = {i: audio_path for i in needed_instrs}

        for instr, p in stem_paths.items():
            log(f"  {instr}: {p.name}")

        # ── 2. Feature extraction (cacheado por instrumento) ──────────────────
        log("\n🎵 Extrayendo features por instrumento ...")
        mert_ext   = MERTExtractor(device=device)
        logmel_ext = LogMelExtractor()

        feat_cache: dict[str, dict] = {}
        for instr in needed_instrs:
            stem = stem_paths.get(instr, audio_path)
            log(f"  {instr} — beats + MERT + LogMel ...")

            beat_info  = BeatEstimator.estimate(stem)
            mert_arr   = mert_ext.extract_per_beat(stem, beat_info["beat_times_s"], beat_info["beat_durations_s"])
            logmel_arr = logmel_ext.extract_per_beat(stem, beat_info["beat_times_s"], beat_info["beat_durations_s"])

            N = min(mert_arr.shape[0], logmel_arr.shape[0], len(beat_info["beat_times_s"]))
            mert_arr   = mert_arr[:N]
            logmel_arr = logmel_arr[:N]

            feat_cache[instr] = {
                "mert":   torch.from_numpy(mert_arr).unsqueeze(0).float().to(device),
                "logmel": torch.from_numpy(logmel_arr).unsqueeze(0).float().to(device),
                "bpm":    torch.tensor([beat_info["bpm_at_beat"][:N]], dtype=torch.float32, device=device),
                "ts_num": torch.tensor([beat_info["time_sig_num_at_beat"][:N]], dtype=torch.long, device=device),
                "ts_den": torch.tensor([beat_info["time_sig_den_at_beat"][:N]], dtype=torch.long, device=device),
                "dur":    torch.tensor([beat_info["beat_durations_s"][:N]], dtype=torch.float32, device=device),
                "mask":   torch.ones(1, N, dtype=torch.bool, device=device),
                "bpm_mean": beat_info["bpm_mean"],
                "N": N,
            }
            log(f"    ✓ {N} beats | BPM≈{beat_info['bpm_mean']:.1f}")

        # ── 3. Autoregressive generation ──────────────────────────────────────
        log(f"\n🤖 Generando {len(targets)} tracks (nucleus T={temperature}, top_p={top_p}) ...")
        generated: dict[tuple[str, int], list[int]] = {}
        model.eval()
        with torch.no_grad():
            for instr, diff_id in targets:
                if cancel_event is not None and cancel_event.is_set():
                    log("⚠ Cancelado por el usuario.")
                    return "Cancelado por el usuario.", None

                feat = feat_cache[instr]
                log(f"  {instr}/{_DIFF_ID_TO_LABEL.get(diff_id, diff_id)} ...")
                tokens = model.generate(
                    mert_embeddings   = feat["mert"],
                    logmel_frames     = feat["logmel"],
                    bpm_at_beat       = feat["bpm"],
                    time_sig_num      = feat["ts_num"],
                    time_sig_den      = feat["ts_den"],
                    beat_duration_s   = feat["dur"],
                    beat_padding_mask = feat["mask"],
                    instrument_id     = _INSTR_KEY_TO_ID[instr],
                    difficulty_id     = diff_id,
                    max_new_tokens    = max_new_tokens,
                    temperature       = temperature,
                    top_k             = 0,
                    top_p             = top_p,
                )
                generated[(instr, diff_id)] = tokens
                log(f"    → {len(tokens)} tokens")

        # ── 4. Decode tokens → combined ChartData ─────────────────────────────
        log("\n📝 Decodificando tokens ...")
        combined = ChartData(resolution=192)
        first_instr = needed_instrs[0]
        bpm_map = BPMMap(resolution=192)
        bpm_map.bpm_events = [BPMEvent(tick=0, bpm=feat_cache[first_instr]["bpm_mean"])]
        combined.bpm_map = bpm_map

        # Keep highest difficulty per instrument (targets are sorted best-first)
        diff_by_instr: dict[str, int] = {}

        for instr, diff_id in targets:
            tokens = generated[(instr, diff_id)]
            instr_bpm = BPMMap(resolution=192)
            instr_bpm.bpm_events = [BPMEvent(tick=0, bpm=feat_cache[instr]["bpm_mean"])]
            chart = decode_tokens(tokens, resolution=192, bpm_map=instr_bpm)

            # Log what the decoder found in each track
            log(f"  [DEBUG] decode_tokens for {instr}/{diff_id}: tracks={list(chart.tracks.keys())}, "
                f"notes_per_track={[len(v) for v in chart.tracks.values()]}")

            notes_for_instr = chart.tracks.get(instr, [])
            track_key = f"{instr}_{diff_id}"
            # Replace track only if this diff is higher priority than existing
            if track_key not in diff_by_instr or _DIFF_PRIORITY.get(diff_id, 99) < _DIFF_PRIORITY.get(diff_by_instr[track_key], 99):
                combined.tracks[track_key]   = notes_for_instr
                combined.specials[track_key] = chart.specials.get(instr, [])
                diff_by_instr[track_key] = diff_id
                log(f"  {instr}/{_DIFF_ID_TO_LABEL.get(diff_id)}: {len(notes_for_instr)} notas")

        # ── 5. Render notes.chart ─────────────────────────────────────────────
        chart_text = render_chart(
            combined,
            bpm=feat_cache[first_instr]["bpm_mean"],
            song_name=song_name or "Unknown",
            artist=artist or "Unknown",
            album=album,
            year=year_int,
            charter="auto-charter",
            diff_by_instr=diff_by_instr,
        )

        # ── 6. Audio → song.ogg ───────────────────────────────────────────────
        log("\n🔊 Procesando audio ...")
        safe_name = "".join(c for c in f"{artist or 'Unknown'} - {song_name or 'Song'}"
                            if c.isalnum() or c in " _-")[:50].strip()
        pkg_dir = tmpdir / safe_name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        ogg_out = pkg_dir / "song.ogg"
        if audio_path.suffix.lower() == ".ogg":
            shutil.copy2(audio_path, ogg_out)
        else:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-c:a", "libvorbis", "-q:a", "6", str(ogg_out)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log(f"  ffmpeg falló — copiando audio original")
                suffix = audio_path.suffix
                ogg_out = pkg_dir / ("song" + suffix)
                shutil.copy2(audio_path, ogg_out)

        audio_filename = ogg_out.name

        # ── 7. song.ini ───────────────────────────────────────────────────────
        try:
            import librosa
            y_mono, sr_m = librosa.load(str(audio_path), sr=22050, mono=True)
            song_length_ms = int(len(y_mono) / sr_m * 1000)
        except Exception:
            song_length_ms = 0

        # Extract best difficulty per base instrument for song.ini
        ini_diffs = {}
        for track_key, diff_id in diff_by_instr.items():
            base = track_key.rsplit("_", 1)[0]
            if base not in ini_diffs or _DIFF_PRIORITY.get(diff_id, 99) < _DIFF_PRIORITY.get(ini_diffs[base], 99):
                ini_diffs[base] = diff_id

        ini_lines = [
            "[Song]",
            f"name = {song_name or 'Unknown'}",
            f"artist = {artist or 'Unknown'}",
        ]
        if album:  ini_lines.append(f"album = {album}")
        if genre:  ini_lines.append(f"genre = {genre}")
        if year_int: ini_lines.append(f"year = {year_int}")
        if song_length_ms: ini_lines.append(f"song_length = {song_length_ms}")
        ini_lines += [
            "charter = auto-charter",
            f"diff_guitar = {ini_diffs.get('guitar', -1)}",
            f"diff_bass = {ini_diffs.get('bass', -1)}",
            f"diff_drums = {ini_diffs.get('drums', -1)}",
            f'MusicStream = "{audio_filename}"',
        ]
        ini_text = "\n".join(ini_lines)

        (pkg_dir / "notes.chart").write_text(chart_text, encoding="utf-8")
        (pkg_dir / "song.ini").write_text(ini_text, encoding="utf-8")

        # ── 7b. Album artwork ───────────────────────────────────────────────────
        if artwork_path and artwork_path.exists():
            shutil.copy2(artwork_path, pkg_dir / "album.png")

        # ── 8. ZIP ────────────────────────────────────────────────────────────
        zip_path = output_dir / f"{safe_name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in pkg_dir.iterdir():
                zf.write(f, arcname=f"{safe_name}/{f.name}")

        size_mb = zip_path.stat().st_size / 1_048_576
        tracks_summary = " | ".join(
            f"{i}({_DIFF_ID_TO_LABEL.get(d, d)})" for i, d in diff_by_instr.items()
        )
        status = (
            f"✓ Generado: '{song_name}' — {tracks_summary}\n"
            f"BPM≈{feat_cache[first_instr]['bpm_mean']:.1f} | "
            f"ZIP: {size_mb:.1f} MB | {len(diff_by_instr)} track(s)"
        )
        log(f"\n{status}")
        return status, zip_path

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log(tb)
        return f"Error: {exc}", None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_app(checkpoint_path: str, device: str = "auto") -> "gr.Blocks":
    import gradio as gr
    import torch
    import threading
    from auto_charter.model.charter_model import AutoCharterModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Cargando modelo desde {checkpoint_path} en {device} ...")
    model = AutoCharterModel.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)
    print(f"Modelo listo ({model.num_parameters():,} parámetros).")

    # Shared state for cancellation
    cancel_event = threading.Event()
    artwork_cache: dict[str, Path] = {}

    def _on_sync(song_name, artist):
        if not song_name or not artist:
            return "", "", "", gr.update(visible=False)
        meta = fetch_metadata_itunes(artist, song_name)
        if meta is None:
            return "", "", "", gr.update(visible=False)
        art_path = Path(tempfile.gettempdir()) / f"autocharter_art_{hash((artist, song_name))}.png"
        downloaded = download_album_art(meta["artwork"], art_path)
        if downloaded:
            artwork_cache["current"] = art_path
            return (
                meta["album"],
                meta["genre"],
                meta["year"],
                gr.update(value=str(art_path), visible=True),
            )
        return meta["album"], meta["genre"], meta["year"], gr.update(visible=False)

    def _on_generate(
        audio_file,
        song_name, artist, album, genre, year,
        instrument_checks, difficulty_checks,
        temperature, top_p, max_new_tokens,
        use_demucs,
        progress=gr.Progress(),
    ):
        cancel_event.clear()
        log_lines: list[str] = []

        def _log(msg: str) -> None:
            log_lines.append(msg)
            try:
                progress(0, desc=msg)
            except Exception:
                pass
            print(msg)

        if audio_file is None:
            return "Por favor sube un archivo de audio.", gr.update(visible=False)

        audio_path = Path(audio_file if isinstance(audio_file, str) else audio_file.name)
        art_path = artwork_cache.get("current")

        status, zip_path = run_pipeline(
            audio_path        = audio_path,
            song_name         = song_name,
            artist            = artist,
            album             = album,
            genre             = genre,
            year              = str(year),
            instrument_labels = instrument_checks,
            difficulty_labels = difficulty_checks,
            model             = model,
            device            = device,
            temperature       = float(temperature),
            top_p             = float(top_p),
            max_new_tokens    = int(max_new_tokens),
            use_demucs        = use_demucs,
            log               = _log,
            cancel_event      = cancel_event,
            artwork_path      = art_path,
        )

        full_status = status + "\n\n── Log ──\n" + "\n".join(log_lines)
        zip_path_str = str(zip_path.resolve()) if zip_path and zip_path.exists() else None
        if zip_path_str:
            return full_status, gr.update(value=zip_path_str, visible=True)
        return full_status, gr.update(visible=False)

    def _on_cancel():
        cancel_event.set()
        return "⚠ Solicitud de cancelación enviada."

    # ── Layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="Auto-Charter", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎸 Auto-Charter\n"
            "Genera un chart de Clone Hero a partir de cualquier archivo de audio.\n"
            "Sube la canción, elige instrumentos y dificultades, y descarga el `.zip`."
        )

        with gr.Row():
            # ── Columna izquierda — entrada ───────────────────────────────────
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Audio (.ogg / .mp3 / .wav)",
                    sources=["upload"],
                    interactive=True,
                )

                gr.Markdown("### Instrumentos")
                instrument_checks = gr.CheckboxGroup(
                    choices=["Guitar", "Bass", "Drums"],
                    value=["Guitar"],
                    label="Seleccionar instrumentos",
                )

                use_demucs_checkbox = gr.Checkbox(
                    value=True,
                    label="Usar Demucs (separar stems por instrumento)",
                    info="Si está desactivado se usa el audio completo para todos los instrumentos",
                )

                gr.Markdown("### Dificultades")
                difficulty_checks = gr.CheckboxGroup(
                    choices=["Easy", "Medium", "Hard", "Expert"],
                    value=["Expert"],
                    label="Generar estas dificultades por instrumento",
                )

                with gr.Accordion("Parámetros de decodificación", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.5, maximum=1.5, value=0.95, step=0.05,
                        label="Temperature (0.95 recomendado)",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.5, maximum=1.0, value=0.92, step=0.01,
                        label="Top-p / Nucleus (0.92 recomendado)",
                    )
                    max_tokens_input = gr.Number(
                        value=8192, label="Max tokens por track", precision=0,
                    )

                with gr.Row():
                    generate_btn = gr.Button("🎵 Generar Chart", variant="primary", size="lg")
                    cancel_btn   = gr.Button("⏹ Cancelar", variant="stop", size="lg")

            # ── Columna derecha — metadatos ───────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Metadatos de la canción")
                with gr.Row():
                    song_name_input = gr.Textbox(label="Nombre de la canción", placeholder="Mi Canción", scale=2)
                    artist_input    = gr.Textbox(label="Artista",              placeholder="Artista", scale=2)
                sync_btn = gr.Button("🔄 Sincronizar", size="sm", scale=1)
                album_input     = gr.Textbox(label="Álbum",                placeholder="Álbum (opcional)")
                genre_input     = gr.Textbox(label="Género",               placeholder="Rock")
                year_input      = gr.Textbox(label="Año",                  placeholder="2024", value="")
                album_art_input = gr.Image(label="Carátula del álbum", interactive=False, visible=False)

                gr.Markdown("### Resultado")
                status_box = gr.Textbox(
                    label="Estado / Log",
                    lines=10,
                    interactive=False,
                    placeholder="El estado aparecerá aquí tras la generación...",
                )
                download_btn = gr.DownloadButton(
                    "⬇ Descargar .zip — arrástralo a la carpeta Songs de Clone Hero",
                    visible=False,
                )

        generate_btn.click(
            fn=_on_generate,
            inputs=[
                audio_input,
                song_name_input, artist_input, album_input, genre_input, year_input,
                instrument_checks, difficulty_checks,
                temperature_slider, top_p_slider, max_tokens_input,
                use_demucs_checkbox,
            ],
            outputs=[status_box, download_btn],
        )

        sync_btn.click(
            fn=_on_sync,
            inputs=[song_name_input, artist_input],
            outputs=[album_input, genre_input, year_input, album_art_input],
        )

        cancel_btn.click(
            fn=_on_cancel,
            inputs=[],
            outputs=[status_box],
        )

        gr.Markdown(
            "---\n"
            "**Notas:**\n"
            "- La separación de stems (Demucs) puede tardar 1–3 min en CPU. Desactívala si prefieres usar el audio completo.\n"
            "- MERT requiere ~4 GB VRAM; en CPU es más lento pero funciona.\n"
            "- Nucleus sampling (T=0.95, top-p=0.92) ofrece la mejor calidad.\n"
            "- El ZIP contiene `notes.chart`, `song.ini` y `song.ogg` en una "
            "carpeta lista para copiar a `Songs/` en Clone Hero."
        )

    return demo


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Directorio del checkpoint (config.json + model.pt)")
@click.option("--port",   default=7860, type=int,  show_default=True)
@click.option("--host",   default="0.0.0.0",       show_default=True)
@click.option("--device", default="auto",           show_default=True,
              help="'auto', 'cuda' o 'cpu'")
@click.option("--share",  is_flag=True,
              help="Crear enlace público de Gradio")
def main(checkpoint, port, host, device, share):
    """Lanzar el demo Gradio multi-instrumento de Auto-Charter."""
    app = build_app(checkpoint_path=checkpoint, device=device)
    app.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
