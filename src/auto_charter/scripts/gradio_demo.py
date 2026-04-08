"""Gradio demo — full pipeline from audio upload to Clone Hero .zip download.

Pipeline:
  1. User uploads audio (.ogg / .mp3 / .wav)
  2. User fills metadata (artist, song name, album, genre, year)
  3. User selects instrument (Guitar / Bass / Drums)
  4. User selects difficulty (Easy / Medium / Hard / Expert)
  5. Backend:
     a. Demucs stem separation → instrument stem
     b. Beat estimation (librosa) → beat_times_s, bpm_at_beat, etc.
     c. MERT embedding extraction per beat
     d. Log-mel extraction per beat
     e. Model inference → token IDs
     f. decode_tokens() → ChartData
     g. render_chart() → notes.chart text
     h. render_ini() → song.ini text
     i. Package .zip: notes.chart + song.ini + audio file
  6. User downloads .zip (ready for Clone Hero)

Launch:
    uv run demo-charter --checkpoint ./checkpoints/run1/best [--port 7860]
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import click


# ── Difficulty / Instrument mappings ─────────────────────────────────────────

_DIFF_LABEL_TO_ID = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2,
    "Expert": 3,
}

_INSTR_LABEL_TO_KEY = {
    "Guitar": "guitar",
    "Bass": "bass",
    "Drums": "drums",
}

_INSTR_KEY_TO_ID = {"guitar": 0, "bass": 1, "drums": 2}


# ── Full generation pipeline ──────────────────────────────────────────────────

def generate_chart(
    audio_file,
    artist: str,
    song_name: str,
    album: str,
    genre: str,
    year: str,
    instrument_label: str,
    difficulty_label: str,
    model,
    device: str,
    progress=None,
) -> tuple[str, str | None]:
    """Run the full pipeline. Returns (status_message, zip_path_or_None)."""
    import numpy as np
    import torch

    from auto_charter.audio.beat_estimator import BeatEstimator
    from auto_charter.audio.logmel import LogMelExtractor
    from auto_charter.audio.mert_extractor import MERTExtractor
    from auto_charter.audio.separator import StemSeparator
    from auto_charter.parsers.chart_renderer import render_chart, render_ini
    from auto_charter.tokenizer.decoder import decode_tokens

    def _prog(frac: float, desc: str) -> None:
        if progress is not None:
            try:
                progress(frac, desc=desc)
            except Exception:
                pass
        print(f"[{int(frac*100):3d}%] {desc}")

    if audio_file is None:
        return "Please upload an audio file.", None

    instr_key = _INSTR_LABEL_TO_KEY.get(instrument_label, "guitar")
    instr_id = _INSTR_KEY_TO_ID[instr_key]
    diff_id = _DIFF_LABEL_TO_ID.get(difficulty_label, 3)
    year_int = int(year) if str(year).strip().isdigit() else 0

    tmpdir = Path(tempfile.mkdtemp(prefix="auto_charter_"))

    try:
        audio_path = Path(audio_file if isinstance(audio_file, str) else audio_file.name)

        # Step 1: Stem separation
        _prog(0.05, "Separating stems with Demucs...")
        stems_dir = tmpdir / "stems"
        try:
            stem_paths = StemSeparator(device=device).separate(
                audio_path, stems_dir, instruments=[instr_key]
            )
            stem_path = stem_paths.get(instr_key, audio_path)
        except Exception as e:
            print(f"Stem separation failed ({e}), using original audio.")
            stem_path = audio_path

        # Step 2: Beat estimation
        _prog(0.20, "Estimating beat grid...")
        beat_info = BeatEstimator.estimate(stem_path)
        beat_times_s = beat_info["beat_times_s"]
        beat_durations_s = beat_info["beat_durations_s"]
        bpm = beat_info["bpm_mean"]
        N = beat_info["num_beats"]

        # Step 3: MERT extraction
        _prog(0.35, f"Extracting MERT embeddings ({N} beats)...")
        mert_arr = MERTExtractor(device=device).extract_per_beat(
            stem_path, beat_times_s, beat_durations_s
        )  # [N, 768]

        # Step 4: LogMel extraction
        _prog(0.50, "Extracting log-mel spectrograms...")
        logmel_arr = LogMelExtractor().extract_per_beat(
            stem_path, beat_times_s, beat_durations_s
        )  # [N, 32, 128]

        # Align lengths (beat detection and feature extraction may differ by 1)
        N_actual = min(mert_arr.shape[0], logmel_arr.shape[0], len(beat_times_s))
        mert_arr = mert_arr[:N_actual]
        logmel_arr = logmel_arr[:N_actual]
        beat_times_s = beat_times_s[:N_actual]
        beat_durations_s = beat_durations_s[:N_actual]
        bpm_at_beat = [bpm] * N_actual
        ts_num = beat_info["time_sig_num_at_beat"][:N_actual]
        ts_den = beat_info["time_sig_den_at_beat"][:N_actual]

        # Step 5: Model inference
        _prog(0.65, "Generating chart tokens (autoregressive)...")
        mert_t = torch.from_numpy(mert_arr).unsqueeze(0).float().to(device)
        logmel_t = torch.from_numpy(logmel_arr).unsqueeze(0).float().to(device)
        bpm_t = torch.tensor([bpm_at_beat], dtype=torch.float32, device=device)
        ts_num_t = torch.tensor([ts_num], dtype=torch.long, device=device)
        ts_den_t = torch.tensor([ts_den], dtype=torch.long, device=device)
        dur_t = torch.tensor([beat_durations_s], dtype=torch.float32, device=device)
        beat_mask_t = torch.ones(1, N_actual, dtype=torch.bool, device=device)

        tokens = model.generate(
            mert_embeddings=mert_t,
            logmel_frames=logmel_t,
            bpm_at_beat=bpm_t,
            time_sig_num=ts_num_t,
            time_sig_den=ts_den_t,
            beat_duration_s=dur_t,
            beat_padding_mask=beat_mask_t,
            instrument_id=instr_id,
            difficulty_id=diff_id,
            max_new_tokens=model.config.max_new_tokens,
            temperature=model.config.temperature,
            top_k=model.config.top_k,
            top_p=model.config.top_p,
        )

        # Step 6: Decode tokens → ChartData
        _prog(0.80, f"Decoding {len(tokens)} tokens...")
        from auto_charter.parsers.sync_track import BPMMap, BPMEvent
        bpm_map = BPMMap(resolution=192)
        bpm_map.bpm_events = [BPMEvent(tick=0, bpm=bpm)]
        chart_data = decode_tokens(tokens, resolution=192, bpm_map=bpm_map)

        # Step 7: Render notes.chart
        _prog(0.88, "Rendering notes.chart...")
        chart_text = render_chart(
            chart_data,
            bpm=bpm,
            song_name=song_name or "Unknown",
            artist=artist or "Unknown",
            album=album,
            year=year_int,
            charter="auto-charter",
        )

        # Step 8: Render song.ini
        song_length_ms = int(len(
            __import__("librosa").load(str(audio_path), sr=22050, mono=True)[0]
        ) / 22050 * 1000)
        ini_text = render_ini(
            song_name=song_name or "Unknown",
            artist=artist or "Unknown",
            album=album,
            genre=genre,
            year=year_int,
            instrument=instr_key,
            difficulty=diff_id,
            song_length_ms=song_length_ms,
        )

        # Step 9: Package .zip
        _prog(0.93, "Packaging .zip...")

        safe_artist = "".join(c for c in (artist or "Unknown") if c.isalnum() or c in " _-")[:30].strip()
        safe_song = "".join(c for c in (song_name or "Unknown") if c.isalnum() or c in " _-")[:30].strip()
        folder_name = f"{safe_artist} - {safe_song}" if safe_artist else safe_song
        zip_path = tmpdir / f"{folder_name}.zip"

        # Copy audio file to tmpdir with song.ogg name
        audio_ext = audio_path.suffix.lower()
        audio_out_name = "song" + audio_ext

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{folder_name}/notes.chart", chart_text)
            zf.writestr(f"{folder_name}/song.ini", ini_text)
            zf.write(audio_path, f"{folder_name}/{audio_out_name}")

        _prog(1.0, "Done!")
        status = (
            f"Generated {len(tokens)} tokens for '{song_name}' by {artist} "
            f"({instrument_label}, {difficulty_label}).\n"
            f"BPM: {bpm:.1f} | Beats: {N_actual} | Package: {zip_path.name}"
        )
        return status, str(zip_path)

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err)
        return f"Error: {e}\n\n{err}", None


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app(checkpoint_path: str, device: str = "auto") -> "gr.Blocks":
    import gradio as gr
    import torch

    from auto_charter.model.charter_model import AutoCharterModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint_path} on {device} ...")
    model = AutoCharterModel.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)
    print(f"Model loaded ({model.num_parameters():,} parameters).")

    def _generate(
        audio_file,
        artist, song_name, album, genre, year,
        instrument_label, difficulty_label,
        progress=gr.Progress(),
    ):
        return generate_chart(
            audio_file=audio_file,
            artist=artist,
            song_name=song_name,
            album=album,
            genre=genre,
            year=year,
            instrument_label=instrument_label,
            difficulty_label=difficulty_label,
            model=model,
            device=device,
            progress=progress,
        )

    with gr.Blocks(title="Auto-Charter", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Auto-Charter\n"
            "Generate a Clone Hero chart from any audio file using a Transformer model.\n"
            "Upload a song, choose your settings, and download the ready-to-play `.zip`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="Upload Audio (.ogg / .mp3 / .wav)",
                    file_types=[".ogg", ".mp3", ".wav"],
                )
                instrument_radio = gr.Radio(
                    choices=["Guitar", "Bass", "Drums"],
                    value="Guitar",
                    label="Instrument",
                )
                difficulty_radio = gr.Radio(
                    choices=["Easy", "Medium", "Hard", "Expert"],
                    value="Expert",
                    label="Difficulty",
                )
                generate_btn = gr.Button("Generate Chart", variant="primary")

            with gr.Column(scale=1):
                artist_input = gr.Textbox(label="Artist", placeholder="e.g. Caos")
                song_name_input = gr.Textbox(label="Song Name", placeholder="e.g. La Planta")
                album_input = gr.Textbox(label="Album", placeholder="e.g. La Vida Gacha")
                genre_input = gr.Textbox(label="Genre", placeholder="e.g. Rock")
                year_input = gr.Textbox(label="Year", placeholder="e.g. 2000", value="")

        status_box = gr.Textbox(
            label="Status",
            lines=4,
            interactive=False,
            placeholder="Status will appear here after generation...",
        )
        download_btn = gr.File(label="Download .zip", visible=False)

        def on_generate(
            audio_file, artist, song_name, album, genre, year,
            instrument, difficulty, progress=gr.Progress()
        ):
            status, zip_path = _generate(
                audio_file, artist, song_name, album, genre, year,
                instrument, difficulty, progress
            )
            if zip_path:
                return status, gr.update(value=zip_path, visible=True)
            return status, gr.update(visible=False)

        generate_btn.click(
            fn=on_generate,
            inputs=[
                audio_input, artist_input, song_name_input, album_input,
                genre_input, year_input, instrument_radio, difficulty_radio,
            ],
            outputs=[status_box, download_btn],
        )

        gr.Markdown(
            "---\n"
            "**Notes:**\n"
            "- Stem separation (Demucs) can take 30–120s on CPU.\n"
            "- MERT extraction requires ~4 GB VRAM on GPU.\n"
            "- The generated chart is for Expert difficulty by default.\n"
            "- The `.zip` contains `notes.chart`, `song.ini`, and the original audio file.\n"
            "  Extract it into your Clone Hero songs folder to play."
        )

    return demo


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Checkpoint directory (contains config.json + model.pt)")
@click.option("--port", default=7860, type=int, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--device", default="auto", show_default=True, help="'auto', 'cuda', or 'cpu'")
@click.option("--share", is_flag=True, help="Create a public Gradio share link")
def main(checkpoint, port, host, device, share):
    """Launch the Auto-Charter Gradio demo."""
    app = build_app(checkpoint_path=checkpoint, device=device)
    app.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
