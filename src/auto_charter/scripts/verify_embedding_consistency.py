"""verify-embedding-consistency — comprueba si shards de distintas fechas tienen
embeddings MERT compatibles (mismo modelo).

Como todos los datos están en un mismo directorio, el script divide los shards
en "antiguos" y "nuevos" por fecha de modificación y compara las distribuciones
de embeddings estadísticamente (no hay canciones repetidas entre shards, así que
no es posible una comparación directa par a par).

Tests que se realizan:
  1. Norma L2 de embeddings — modelos distintos producen normas sistemáticamente
     diferentes.
  2. Media del embedding — el vector promedio debería estar en una región similar
     del espacio si el modelo es el mismo.
  3. Similitud coseno entre centroide de viejos y centroide de nuevos — debería
     ser ≈ 1.0 si el modelo es el mismo.
  4. KS-test sobre la distribución de normas — p-value bajo indica distribuciones
     estadísticamente distintas.

Uso:
    python -m auto_charter.scripts.verify_embedding_consistency \\
        --dataset-dir  F:/CloneCharter_converted \\
        --cutoff       2026-04-10           # shards < fecha = viejos
        --sample-per-group 500             # embeddings a muestrear por grupo
        --instrument   guitar              # opcional
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, date
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)


# ── Utilidades de carga (sin cargar shards enteros en RAM) ────────────────────


def _find_shards(dataset_dir: Path) -> list[Path]:
    for subdir in [dataset_dir / "shards", dataset_dir]:
        shards = sorted(subdir.glob("*.parquet"))
        if shards:
            return shards
    raise FileNotFoundError(
        f"No se encontraron archivos .parquet en {dataset_dir}.\n"
        "Esperado: subdirectorio 'shards/' o archivos .parquet en la raíz."
    )


def _split_by_date(shards: list[Path], cutoff: date) -> tuple[list[Path], list[Path]]:
    """Divide shards en (antiguos, nuevos) según su fecha de modificación."""
    old, new = [], []
    for p in shards:
        mtime = date.fromtimestamp(os.path.getmtime(str(p)))
        if mtime < cutoff:
            old.append(p)
        else:
            new.append(p)
    return old, new


def _sample_embeddings(
    shards: list[Path],
    n: int,
    instrument: str | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lee embeddings de shards aleatorios hasta obtener n vectores [n, 768].

    Lee shard a shard y solo carga las columnas necesarias para no saturar RAM.
    """
    import pyarrow.parquet as pq

    cols = ["mert_embeddings"]
    if instrument:
        cols.append("instrument")

    vectors: list[np.ndarray] = []
    shard_order = rng.permutation(len(shards)).tolist()

    for shard_idx in shard_order:
        if len(vectors) >= n:
            break

        shard_path = shards[shard_idx]
        try:
            table = pq.read_table(str(shard_path), columns=cols)
        except Exception as e:
            logger.warning("Error leyendo %s: %s", shard_path.name, e)
            continue

        # Filtrar por instrumento si se pidió
        if instrument:
            insts = table.column("instrument").to_pylist()
            mask = [i == instrument for i in insts]
            table = table.filter(np.array(mask, dtype=bool))

        emb_col = table.column("mert_embeddings").to_pylist()
        for emb in emb_col:
            if not emb:
                continue
            arr = np.array(emb, dtype=np.float32)  # [num_beats, 768]
            if arr.ndim != 2 or arr.shape[1] != 768:
                continue
            # Usar la media de beats como representante del track (1 vector por fila)
            vectors.append(arr.mean(axis=0))
            if len(vectors) >= n:
                break

    if not vectors:
        return np.empty((0, 768), dtype=np.float32)

    return np.stack(vectors[:n])


# ── Métricas de comparación de distribuciones ─────────────────────────────────


def _norm_stats(mat: np.ndarray) -> dict:
    norms = np.linalg.norm(mat, axis=1)
    return {
        "mean": float(norms.mean()),
        "std":  float(norms.std()),
        "p5":   float(np.percentile(norms, 5)),
        "p95":  float(np.percentile(norms, 95)),
    }


def _centroid_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre los centroides de dos conjuntos de vectores."""
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    return float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-8))


def _ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """KS-test sobre las normas de dos grupos. Devuelve (statistic, p_value)."""
    from scipy import stats
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    result = stats.ks_2samp(norms_a, norms_b)
    return float(result.statistic), float(result.pvalue)


def _mean_pairwise_cosine(a: np.ndarray, b: np.ndarray, max_pairs: int = 1000) -> float:
    """Similitud coseno media entre pares aleatorios de los dos grupos."""
    rng = np.random.default_rng(0)
    n = min(max_pairs, len(a), len(b))
    ia = rng.choice(len(a), size=n, replace=False)
    ib = rng.choice(len(b), size=n, replace=False)
    va = a[ia] / (np.linalg.norm(a[ia], axis=1, keepdims=True) + 1e-8)
    vb = b[ib] / (np.linalg.norm(b[ib], axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(va * vb, axis=1)))


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--dataset-dir", "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directorio que contiene los shards (o subdirectorio 'shards/').",
)
@click.option(
    "--cutoff",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Fecha de corte (YYYY-MM-DD). Shards anteriores = viejos; posteriores/iguales = nuevos.",
)
@click.option(
    "--sample-per-group", "-n",
    default=500,
    type=int,
    show_default=True,
    help="Número de vectores de embedding a muestrear por grupo.",
)
@click.option(
    "--instrument",
    default=None,
    help="Filtrar por instrumento (guitar/bass/drums). Por defecto usa todos.",
)
@click.option(
    "--no-ks",
    is_flag=True,
    default=False,
    help="Saltar el KS-test (requiere scipy).",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    show_default=True,
)
def main(
    dataset_dir: str,
    cutoff: datetime,
    sample_per_group: int,
    instrument: str | None,
    no_ks: bool,
    log_level: str,
) -> None:
    """Compara distribuciones de embeddings MERT entre shards viejos y nuevos.

    Divide los shards por fecha de modificación y verifica si pertenecen al mismo
    espacio de embeddings (mismo modelo). Lee shard a shard para no saturar la RAM.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
        _RICH = True
    except ImportError:
        console = None
        _RICH = False

    def echo(msg: str) -> None:
        if _RICH:
            console.print(msg)
        else:
            click.echo(msg)

    cutoff_date = cutoff.date()

    # ── Localizar y dividir shards ─────────────────────────────────────────────
    try:
        all_shards = _find_shards(Path(dataset_dir))
    except FileNotFoundError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    shards_old, shards_new = _split_by_date(all_shards, cutoff_date)

    echo(f"\n[bold]Dataset:[/bold] {dataset_dir}" if _RICH else f"Dataset: {dataset_dir}")
    echo(f"[bold]Corte:[/bold]    {cutoff_date}  (shards anteriores = viejos, >= fecha = nuevos)" if _RICH
         else f"Corte:    {cutoff_date}")
    echo("")
    echo(f"  Shards totales : {len(all_shards)}")
    echo(f"  Shards viejos  : {len(shards_old)}")
    echo(f"  Shards nuevos  : {len(shards_new)}")

    if not shards_old or not shards_new:
        echo(
            "\n[bold red]ERROR:[/bold red] Uno de los grupos está vacío. Ajusta --cutoff." if _RICH
            else "\nERROR: Uno de los grupos está vacío. Ajusta --cutoff."
        )
        # Mostrar fechas disponibles para orientar al usuario
        dates = sorted({date.fromtimestamp(os.path.getmtime(str(p))) for p in all_shards})
        echo(f"  Fechas de modificación encontradas: {', '.join(str(d) for d in dates)}")
        sys.exit(1)

    echo(f"\nMuestreando hasta {sample_per_group} vectores por grupo...")
    rng = np.random.default_rng(42)

    vec_old = _sample_embeddings(shards_old, sample_per_group, instrument, rng)
    vec_new = _sample_embeddings(shards_new, sample_per_group, instrument, rng)

    echo(f"  Viejos : {len(vec_old)} vectores obtenidos")
    echo(f"  Nuevos : {len(vec_new)} vectores obtenidos\n")

    if len(vec_old) == 0 or len(vec_new) == 0:
        echo("ERROR: No se pudieron extraer embeddings de uno de los grupos.")
        sys.exit(1)

    # ── Calcular métricas ──────────────────────────────────────────────────────
    stats_old = _norm_stats(vec_old)
    stats_new = _norm_stats(vec_new)
    centroid_cos = _centroid_cosine(vec_old, vec_new)
    cross_cos = _mean_pairwise_cosine(vec_old, vec_new)

    ks_stat, ks_pval = None, None
    if not no_ks:
        try:
            ks_stat, ks_pval = _ks_test(vec_old, vec_new)
        except ImportError:
            echo("[yellow]scipy no disponible — KS-test omitido. pip install scipy[/yellow]" if _RICH
                 else "scipy no disponible — KS-test omitido.")

    # ── Mostrar resultados ─────────────────────────────────────────────────────
    if _RICH:
        table = Table(title="Comparación de distribuciones de embeddings MERT", show_header=True)
        table.add_column("Métrica",        style="cyan", no_wrap=True)
        table.add_column("Viejos",         justify="right")
        table.add_column("Nuevos",         justify="right")
        table.add_column("Diferencia",     justify="right")

        def diff(a, b):
            d = abs(a - b)
            color = "green" if d < 0.5 else ("yellow" if d < 2.0 else "red")
            return f"[{color}]{d:.4f}[/{color}]"

        table.add_row("Norma L2 media",
                      f"{stats_old['mean']:.4f}", f"{stats_new['mean']:.4f}",
                      diff(stats_old['mean'], stats_new['mean']))
        table.add_row("Norma L2 std",
                      f"{stats_old['std']:.4f}", f"{stats_new['std']:.4f}",
                      diff(stats_old['std'], stats_new['std']))
        table.add_row("Norma L2 p5",
                      f"{stats_old['p5']:.4f}", f"{stats_new['p5']:.4f}",
                      diff(stats_old['p5'], stats_new['p5']))
        table.add_row("Norma L2 p95",
                      f"{stats_old['p95']:.4f}", f"{stats_new['p95']:.4f}",
                      diff(stats_old['p95'], stats_new['p95']))
        table.add_section()
        table.add_row("Similitud coseno (centroides)", f"{centroid_cos:.6f}", "", "")
        table.add_row("Similitud coseno cruzada media", f"{cross_cos:.6f}", "", "")
        if ks_stat is not None:
            ks_color = "green" if ks_pval > 0.05 else "red"
            table.add_row(
                "KS-test (normas)",
                f"stat={ks_stat:.4f}",
                f"[{ks_color}]p={ks_pval:.4f}[/{ks_color}]",
                "[green]mismo dist.[/green]" if ks_pval > 0.05 else "[red]dists. distintas[/red]"
            )
        console.print(table)
    else:
        click.echo("\n── Normas L2 ────────────────────────────────────────────────")
        click.echo(f"  {'Métrica':<20} {'Viejos':>10} {'Nuevos':>10} {'|Dif|':>10}")
        click.echo(f"  {'Media':<20} {stats_old['mean']:>10.4f} {stats_new['mean']:>10.4f} {abs(stats_old['mean']-stats_new['mean']):>10.4f}")
        click.echo(f"  {'Std':<20} {stats_old['std']:>10.4f} {stats_new['std']:>10.4f} {abs(stats_old['std']-stats_new['std']):>10.4f}")
        click.echo(f"  {'P5':<20} {stats_old['p5']:>10.4f} {stats_new['p5']:>10.4f} {abs(stats_old['p5']-stats_new['p5']):>10.4f}")
        click.echo(f"  {'P95':<20} {stats_old['p95']:>10.4f} {stats_new['p95']:>10.4f} {abs(stats_old['p95']-stats_new['p95']):>10.4f}")
        click.echo(f"\n  Similitud coseno (centroides) : {centroid_cos:.6f}")
        click.echo(f"  Similitud coseno cruzada media: {cross_cos:.6f}")
        if ks_stat is not None:
            click.echo(f"  KS-test normas: stat={ks_stat:.4f}  p={ks_pval:.4f}  "
                       f"({'mismo dist.' if ks_pval > 0.05 else 'DISTS. DISTINTAS'})")
        click.echo("────────────────────────────────────────────────────────────")

    # ── Veredicto ─────────────────────────────────────────────────────────────
    # Heurísticas basadas en propiedades conocidas de MERT-95M vs 330M:
    # - centroid_cos < 0.85 → modelos claramente distintos
    # - norma media muy diferente (>2.0) → modelos distintos
    # - KS p < 0.001 + centroid_cos < 0.90 → incompatibles
    norm_diff = abs(stats_old["mean"] - stats_new["mean"])
    incompatible_signals = sum([
        centroid_cos < 0.85,
        norm_diff > 2.0,
        (ks_pval is not None and ks_pval < 0.001 and centroid_cos < 0.92),
    ])

    if centroid_cos >= 0.97 and norm_diff < 0.5:
        verdict = "COMPATIBLES — Los embeddings provienen casi seguro del mismo modelo."
        color = "bold green"
        ok = True
    elif centroid_cos >= 0.90 and norm_diff < 1.5:
        verdict = (
            f"PROBABLEMENTE COMPATIBLES — Similitud centroide {centroid_cos:.4f}, "
            f"diferencia de norma {norm_diff:.4f}. Pequeñas diferencias en preprocesado o chunk size."
        )
        color = "green"
        ok = True
    elif incompatible_signals >= 2:
        verdict = (
            f"INCOMPATIBLES — {incompatible_signals}/3 señales de incompatibilidad. "
            f"Similitud centroide {centroid_cos:.4f}, diferencia norma {norm_diff:.4f}. "
            "Muy probable que sean modelos distintos (ej. MERT-95M vs 330M)."
        )
        color = "bold red"
        ok = False
    else:
        verdict = (
            f"DUDOSO — Similitud centroide {centroid_cos:.4f}, diferencia norma {norm_diff:.4f}. "
            "Podría ser el mismo modelo con cambios en el pipeline de audio."
        )
        color = "bold yellow"
        ok = False

    if _RICH:
        console.print(Panel(verdict, style=color, expand=False))
    else:
        click.echo(f"\n{verdict}\n")

    # Guía de interpretación
    guide = (
        "\n[dim]Guia de interpretacion:[/dim]\n"
        "[dim]  Similitud centroide >= 0.97  -> mismo modelo[/dim]\n"
        "[dim]  Similitud centroide 0.85-0.97 -> dudoso[/dim]\n"
        "[dim]  Similitud centroide < 0.85  -> modelos distintos[/dim]\n"
        "[dim]  Diferencia norma > 2.0     -> modelos distintos[/dim]\n"
        "[dim]  KS p < 0.001              -> distribuciones distintas[/dim]"
    )
    if _RICH:
        console.print(guide)
    else:
        click.echo(
            "\nGuía: centroide>=0.97 mismo modelo | 0.85-0.97 dudoso | <0.85 distintos\n"
            "      norma diff>2.0 distintos | KS p<0.001 distribuciones distintas"
        )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
