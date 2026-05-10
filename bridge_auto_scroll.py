#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         Bridge Auto-Scroll & Download — Dataset Helper       ║
║  Encuentra la ventana de Bridge, hace scroll automático y    ║
║  pulsa el botón de descarga cuando hay suficientes canciones ║
╚══════════════════════════════════════════════════════════════╝

Requisitos:
    pip install pyautogui pygetwindow Pillow

Uso:
    python bridge_auto_scroll.py                 # modo normal
    python bridge_auto_scroll.py --calibrate     # modo calibración (para ajustar coords)
    python bridge_auto_scroll.py --songs 500     # descargar 500 canciones
"""

import argparse
import time
import sys
import logging

import pyautogui
import pygetwindow as gw

# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  ← AJUSTA ESTOS VALORES SI ES NECESARIO
# ══════════════════════════════════════════════════════════════

# Nombre parcial de la ventana de Bridge (insensible a mayúsculas)
WINDOW_TITLE = "Bridge"

# Cuántas canciones cargar antes de pulsar descarga
TARGET_SONGS = 1000000

# Velocidad del scroll: pausa entre cada scroll (segundos)
SCROLL_PAUSE = 0.0          # sin pausa entre scrolls

# Cuántos "ticks" de scroll por paso
SCROLL_AMOUNT = 150          # scroll muy agresivo por paso

# Cuántas veces scrollear sin encontrar canciones nuevas antes de parar
MAX_STALE_ROUNDS = 8

# Pausa antes de pulsar el botón de descarga (segundos)
PRE_DOWNLOAD_PAUSE = 1.5

# Pausa entre descargas individuales si el botón es por canción (segundos)
PER_SONG_PAUSE = 0.15

# ── Offsets relativos a la ventana Bridge (px desde esquina superior-izquierda) ──
# Estos valores funcionan para la distribución estándar de Bridge.
# Si tu ventana tiene un tamaño diferente, usa --calibrate para ajustarlos.

# Centro del panel de lista de canciones (donde se hace scroll)
SONG_LIST_REL_X = 0.35      # 35% del ancho de la ventana
SONG_LIST_REL_Y = 0.50      # 50% de la altura (centro vertical)

# Botón "Download All" o "Download Selected" (esquina inferior derecha aprox.)
DOWNLOAD_BTN_REL_X = 0.85
DOWNLOAD_BTN_REL_Y = 0.92

# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge_scroll")

pyautogui.FAILSAFE = True   # mover ratón a esquina superior-izq = parada de emergencia
pyautogui.PAUSE    = 0.0    # sin pausa global


# ══════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════

def find_bridge_window():
    """Busca la ventana de Bridge entre todas las ventanas abiertas."""
    all_windows = gw.getAllWindows()
    matches = [w for w in all_windows if WINDOW_TITLE.lower() in w.title.lower()]

    if not matches:
        log.error(
            f"No se encontró ninguna ventana con '{WINDOW_TITLE}' en el título.\n"
            f"Ventanas abiertas: {[w.title for w in all_windows]}"
        )
        sys.exit(1)

    if len(matches) > 1:
        log.warning(f"Múltiples ventanas coinciden: {[w.title for w in matches]}. Usando la primera.")

    win = matches[0]
    log.info(f"Ventana encontrada: '{win.title}' en ({win.left}, {win.top}) {win.width}×{win.height}px")
    return win


def bring_to_front(win):
    """Trae la ventana de Bridge al frente."""
    try:
        win.activate()
    except Exception:
        pass
    time.sleep(0.5)


def abs_coords(win, rel_x: float, rel_y: float) -> tuple[int, int]:
    """Convierte coordenadas relativas a absolutas según la ventana."""
    x = int(win.left + win.width  * rel_x)
    y = int(win.top  + win.height * rel_y)
    return x, y


def scroll_song_list(win, amount: int = SCROLL_AMOUNT):
    """Hace scroll hacia abajo en el panel de canciones."""
    x, y = abs_coords(win, SONG_LIST_REL_X, SONG_LIST_REL_Y)
    pyautogui.moveTo(x, y, duration=0)
    pyautogui.scroll(-amount)  # negativo = hacia abajo


def click_download(win):
    """Pulsa el botón de descarga."""
    x, y = abs_coords(win, DOWNLOAD_BTN_REL_X, DOWNLOAD_BTN_REL_Y)
    log.info(f"Pulsando botón de descarga en ({x}, {y})")
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.click()


def count_visible_songs(win) -> int:
    """
    Intenta contar canciones visibles haciendo una captura de pantalla de la ventana
    y contando líneas oscuras/separadores. Es una heurística — puede no ser exacta.
    Si prefieres, puedes desactivar esto y fiarte solo de los rondas de scroll.
    """
    try:
        from PIL import ImageGrab, ImageFilter
        import numpy as np

        shot = ImageGrab.grab(bbox=(win.left, win.top, win.right, win.bottom))
        # Convertir a escala de grises y buscar líneas separadoras horizontales
        gray = shot.convert("L")
        arr  = list(gray.getdata())
        width, height = gray.size
        rows = [arr[i * width:(i + 1) * width] for i in range(height)]

        # Una "línea separadora" es una fila con muchos píxeles del mismo tono
        separators = 0
        for row in rows[50:]:  # ignorar barra de título
            avg = sum(row) / len(row)
            spread = max(row) - min(row)
            if spread < 15 and 50 < avg < 220:  # línea uniforme
                separators += 1

        # Muy heurístico: ~20-30px por canción
        estimated = separators // 2
        return estimated

    except ImportError:
        return -1  # numpy/PIL no disponible, no contar


# ══════════════════════════════════════════════════════════════
#  MODO CALIBRACIÓN
# ══════════════════════════════════════════════════════════════

def calibrate_mode():
    """
    Modo interactivo para ajustar las coordenadas relativas.
    Mueve el ratón a la posición deseada y pulsa ENTER para registrarla.
    """
    win = find_bridge_window()
    bring_to_front(win)

    print("\n╔══════════════════════════════════════════════╗")
    print("║         MODO CALIBRACIÓN                    ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ Mueve el ratón a la posición indicada y     ║")
    print("║ pulsa ENTER para registrar las coordenadas. ║")
    print("╚══════════════════════════════════════════════╝\n")

    def get_position(label: str) -> tuple[float, float]:
        input(f"  ▶ Mueve el ratón al {label} y pulsa ENTER...")
        mx, my = pyautogui.position()
        rel_x = (mx - win.left) / win.width
        rel_y = (my - win.top)  / win.height
        print(f"    → abs=({mx}, {my})  rel=({rel_x:.3f}, {rel_y:.3f})")
        return rel_x, rel_y

    list_x, list_y = get_position("CENTRO de la lista de canciones (para hacer scroll)")
    btn_x,  btn_y  = get_position("BOTÓN DE DESCARGA")

    print("\n── Copia estos valores en CONFIGURACIÓN del script ──")
    print(f"SONG_LIST_REL_X  = {list_x:.3f}")
    print(f"SONG_LIST_REL_Y  = {list_y:.3f}")
    print(f"DOWNLOAD_BTN_REL_X = {btn_x:.3f}")
    print(f"DOWNLOAD_BTN_REL_Y = {btn_y:.3f}")
    print("─────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL DE SCROLL
# ══════════════════════════════════════════════════════════════

def scroll_and_download(target_songs: int):
    win = find_bridge_window()
    bring_to_front(win)

    log.info(f"Objetivo: {target_songs:,} canciones")
    log.info("Iniciando scroll… (mueve el ratón a la esquina superior-izquierda para detener)")
    time.sleep(1.0)

    scroll_rounds    = 0
    stale_rounds     = 0
    prev_scroll_pos  = 0  # aproximación: rounds × SCROLL_AMOUNT

    try:
        while True:
            # ── Refresca las coords por si la ventana se movió/redimensionó ──
            wins = gw.getWindowsWithTitle(win.title)
            if wins:
                win = wins[0]

            scroll_song_list(win)
            time.sleep(SCROLL_PAUSE)
            scroll_rounds += 1

            cur_pos = scroll_rounds * SCROLL_AMOUNT

            # Detección de "fin de lista" (scroll no avanzó nada nuevo)
            if cur_pos == prev_scroll_pos:
                stale_rounds += 1
            else:
                stale_rounds    = 0
                prev_scroll_pos = cur_pos

            if stale_rounds >= MAX_STALE_ROUNDS:
                log.info("Scroll detenido: no hay más contenido nuevo.")
                break

            # ── Estimación de progreso cada 25 rondas ──
            if scroll_rounds % 25 == 0:
                log.info(f"  Ronda {scroll_rounds} — scroll pos ≈ {cur_pos}")

            # ── Comprobar si alcanzamos el objetivo ──
            # La estimación por screenshot es opcional; si no funciona,
            # usamos las rondas de scroll como proxy.
            songs_est = count_visible_songs(win)
            if songs_est > 0:
                if songs_est >= target_songs:
                    log.info(f"¡Objetivo alcanzado! ~{songs_est} canciones visibles.")
                    break
            else:
                # Sin estimación visual: usar heurística de rondas
                # ~10-15 canciones por ronda de scroll (ajusta según Bridge)
                SONGS_PER_ROUND = 12
                if scroll_rounds * SONGS_PER_ROUND >= target_songs:
                    log.info(
                        f"Objetivo estimado alcanzado tras {scroll_rounds} rondas "
                        f"(~{scroll_rounds * SONGS_PER_ROUND} canciones)."
                    )
                    break

    except pyautogui.FailSafeException:
        log.warning("¡FailSafe activado! Parando scroll.")
        return

    # ── Descargar ──
    log.info(f"Esperando {PRE_DOWNLOAD_PAUSE}s antes de pulsar descarga…")
    time.sleep(PRE_DOWNLOAD_PAUSE)

    try:
        click_download(win)
        log.info("Botón de descarga pulsado. Bridge debería iniciar las descargas.")
        log.info("Si Bridge pide confirmación, acéptala manualmente.")
    except pyautogui.FailSafeException:
        log.warning("FailSafe activado durante la descarga.")


# ══════════════════════════════════════════════════════════════
#  ENTRADA
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Automatiza el scroll y descarga masiva en Bridge (Clone Hero)."
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Modo calibración: muestra coordenadas relativas para ajustar el script."
    )
    parser.add_argument(
        "--songs", type=int, default=TARGET_SONGS,
        help=f"Número de canciones a cargar antes de descargar (default: {TARGET_SONGS})."
    )
    parser.add_argument(
        "--window", type=str, default=WINDOW_TITLE,
        help=f"Título (parcial) de la ventana de Bridge (default: '{WINDOW_TITLE}')."
    )
    args = parser.parse_args()

    if args.window != WINDOW_TITLE:
        globals()["WINDOW_TITLE"] = args.window

    if args.calibrate:
        calibrate_mode()
    else:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║   Bridge Auto-Scroll — Dataset Helper                   ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║  Objetivo  : {args.songs:,} canciones".ljust(59) + "║")
        print(f"║  Ventana   : '{WINDOW_TITLE}'".ljust(59) + "║")
        print("║  ⚠  Mueve el ratón a la esquina sup-izq para detener    ║")
        print("╚══════════════════════════════════════════════════════════╝\n")
        print("Tienes 3 segundos para cambiar a Bridge…")
        for i in range(3, 0, -1):
            print(f"  {i}…")
            time.sleep(1)

        scroll_and_download(args.songs)


if __name__ == "__main__":
    main()
