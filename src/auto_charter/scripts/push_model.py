"""Sube el checkpoint de AutoCharterModel a Hugging Face Hub.

Uso:
    python src/auto_charter/scripts/push_model.py \
        --checkpoint checkpoints/run1/best \
        --repo thejorseman/CloneCharter \
        [--token hf_xxx]  # o exporta HF_TOKEN
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import tempfile

import click
import torch
from huggingface_hub import HfApi, CommitOperationAdd
from safetensors.torch import save_file as st_save

_MODEL_CARD = """\
---
license: mit
language:
  - en
tags:
  - audio
  - music
  - clone-hero
  - chart-generation
  - encoder-decoder
  - transformer
library_name: auto-charter
---

# AutoCharterModel — Clone Hero Chart Generator

Encoder-decoder Transformer that generates Clone Hero `.chart` files from raw audio.

## Model description

AutoCharterModel takes per-beat audio features (MERT embeddings + Log-Mel spectrogram)
and autoregressively generates a tokenised chart for Guitar, Bass and/or Drums at any
difficulty level (Easy → Expert+++).

### Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | 8,038,017 |
| `d_model` | 256 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 8 |
| FFN dim | 512 |
| Dropout | 0.2 |
| Vocab size | 187 |
| Max sequence length | 8,192 tokens |
| Max beats | 1,024 |
| MERT input dim | 1,024 |
| Log-Mel frames | 32 × 128 |

### Input features (per beat)
- **MERT embeddings** — [N, 1024] from [m-a-p/MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M)
- **Log-Mel spectrogram** — [N, 32, 128] (22 050 Hz, 128 mels)
- **BPM**, time signature numerator/denominator, beat duration (scalar per beat)
- **Instrument ID** — guitar=0, bass=1, drums=2
- **Difficulty ID** — Easy=0, Medium=1, Hard=2, Expert=3, Expert+=4 …

### Vocabulary (187 tokens)

```
PAD=0  BOS=1  EOS=2  UNK=3
BEAT_BOUNDARY=4  MEASURE_START=5
INSTR_GUITAR=6  INSTR_BASS=7  INSTR_DRUMS=8
WAIT(1-48)=9-56
GUITAR_NOTE=57-87  MOD_HOPO=88  MOD_TAP=89  MOD_OPEN=90  MOD_FORCE_STRUM=91
DRUM_NOTE=92-122
SUS(0-59)=123-182
STAR_POWER_ON=183  STAR_POWER_OFF=184  SOLO_ON=185  SOLO_OFF=186
```

## Training

- **Dataset**: ~42,600 Clone Hero charts (Guitar, Bass, Drums)
- **Optimiser**: AdamW (lr=3e-4, weight decay=0.01, cosine schedule)
- **Best validation loss**: 1.0534 at step 145,188

## Intended use

Use the [auto-charter](https://github.com/thejorseman/CloneCharter) pipeline to generate
charts for new songs:

```bash
python src/auto_charter/scripts/gradio_multigen.py \\\\
    --checkpoint path/to/checkpoint \\\\
    --port 7860
```

Or programmatically:

```python
from auto_charter.model.charter_model import AutoCharterModel
model = AutoCharterModel.from_pretrained("thejorseman/CloneCharter")
```

## License

MIT
"""


def _to_safetensors(model_pt: Path, out: Path) -> None:
    """Convierte model.pt (pickle) a model.safetensors.

    Clona tensores antes de guardar para romper el aliasing por weight tying
    (token_emb.weight / lm_head.weight comparten storage en el modelo).
    Los valores son idénticos en el momento de guardar, por lo que el modelo
    cargado tiene comportamiento equivalente.
    """
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    # clone() + contiguous() rompe el shared-memory sin alterar los valores
    state = {k: v.clone().contiguous() for k, v in state.items()}
    st_save(state, str(out))


def push(checkpoint: Path, repo_id: str, token: str | None) -> None:
    api = HfApi(token=token)

    ops = []

    # ── config.json ──────────────────────────────────────────────────────────
    for fname in ("config.json", "trainer_state.json"):
        fpath = checkpoint / fname
        if fpath.exists():
            ops.append(CommitOperationAdd(
                path_in_repo=fname,
                path_or_fileobj=str(fpath),
            ))
            print(f"  Preparando: {fname} ({fpath.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  Advertencia: {fname} no encontrado, omitiendo")

    # ── model.pt → model.safetensors ─────────────────────────────────────────
    model_pt = checkpoint / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"No se encontró {model_pt}")

    with tempfile.TemporaryDirectory() as tmp:
        st_path = Path(tmp) / "model.safetensors"
        print(f"  Convirtiendo model.pt → model.safetensors ...")
        _to_safetensors(model_pt, st_path)
        size_mb = st_path.stat().st_size / 1_048_576
        print(f"  model.safetensors: {size_mb:.1f} MB")

        ops.append(CommitOperationAdd(
            path_in_repo="model.safetensors",
            path_or_fileobj=st_path.read_bytes(),
        ))

        readme_bytes = _MODEL_CARD.encode("utf-8")
        ops.append(CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme_bytes,
        ))
        print("  Preparando: README.md (model card)")

        print(f"\nSubiendo {len(ops)} archivos a {repo_id} ...")
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=ops,
            commit_message="Upload AutoCharterModel checkpoint (run1/best)",
        )
    print(f"\n✓ Subida completada: https://huggingface.co/{repo_id}")


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Directorio del checkpoint (config.json + model.pt)")
@click.option("--repo", default="thejorseman/CloneCharter", show_default=True,
              help="ID del repositorio en Hugging Face Hub")
@click.option("--token", default=None, envvar="HF_TOKEN",
              help="Token de HF (o exportar HF_TOKEN)")
def main(checkpoint: str, repo: str, token: str | None) -> None:
    """Sube el checkpoint de AutoCharterModel a Hugging Face Hub."""
    push(Path(checkpoint), repo, token)


if __name__ == "__main__":
    main()
