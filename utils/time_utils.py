import numpy as np

def song_duration_to_ticks_complete(bpm, duration_seconds, resolution=480, offset=0):
    """
    Función completa que procesa una canción basándose en su duración en segundos
    tomando en cuenta el offset:
    1. Calcula beats totales desde la duración
    2. Convierte BPM a tiempos en segundos
    3. Interpola medios beats
    4. Convierte a ticks aplicando el offset
    
    Args:
        bpm: Beats por minuto
        duration_seconds: Duración total de la canción en segundos (ej: 180)
        resolution: Resolución MIDI (por defecto 480)
        offset: Offset en segundos (por defecto 0) - tiempo de inicio
    
    Returns:
        Diccionario con información completa de la canción
    """
    # Calcular información básica
    seconds_per_beat = 60 / bpm
    total_beats = duration_seconds / seconds_per_beat
    
    # Paso 1: Crear array de tiempos para cada beat (sin offset)
    beat_times = np.arange(0, duration_seconds + seconds_per_beat, seconds_per_beat)
    beat_times = beat_times[beat_times <= duration_seconds]
    
    # Paso 2: Interpolar medios beats (sin offset)
    half_beat_interval = seconds_per_beat / 2
    half_beat_times = np.arange(0, duration_seconds + half_beat_interval, half_beat_interval)
    half_beat_times = half_beat_times[half_beat_times <= duration_seconds]
    
    # Paso 3: Convertir a ticks APLICANDO EL OFFSET
    # Fórmula: Ticks = ((Tiempo - Offset) / (60 / BPM)) * Resolution
    half_ticks = ((half_beat_times - offset) / (60 / bpm)) * resolution
    ticks = ((beat_times - offset) / (60 / bpm)) * resolution


    # Tiempos reales con offset aplicado
    real_times_with_offset = half_beat_times + offset
    
    return {
        'duration_seconds': duration_seconds,
        'offset': offset,
        'total_beats': total_beats,
        'seconds_per_beat': seconds_per_beat,
        'beat_times': beat_times,
        'half_beat_times': half_beat_times,
        'real_times_with_offset': real_times_with_offset,
        'half_ticks': half_ticks,
        'ticks': ticks,  # Convertir a enteros
        #'total_ticks': half_ticks[-1] if len(half_ticks) > 0 else 0
    }

def calculate_ticks_with_offset(times_seconds, bpm, resolution=480, offset=0):
    """
    Convierte tiempos en segundos a ticks aplicando offset.
    
    Args:
        times_seconds: Array de tiempos en segundos
        bpm: Beats por minuto
        resolution: Resolución MIDI
        offset: Offset en segundos
    
    Returns:
        Array de ticks con offset aplicado
    """
    # Fórmula: Ticks = ((Tiempo - Offset) / (60 / BPM)) * Resolution
    ticks = ((times_seconds - offset) / (60 / bpm)) * resolution
    return ticks

def ticks_to_seconds_with_offset(ticks, bpm, resolution=480, offset=0):
    """
    Convierte ticks a segundos aplicando offset (función inversa).
    
    Args:
        ticks: Array de ticks
        bpm: Beats por minuto
        resolution: Resolución MIDI
        offset: Offset en segundos
    
    Returns:
        Array de tiempos en segundos
    """
    # Fórmula inversa: Tiempo = (Ticks / Resolution) * (60 / BPM) + Offset
    times = (ticks / resolution) * (60 / bpm) + offset
    return times

"""
# Ejemplo 1: Sin offset (offset = 0)
duration_seconds = 180
bpm = 120
resolution = 192
offset = 0

result1 = song_duration_to_ticks_complete(bpm, duration_seconds, resolution, offset)

print(f"Duración: {duration_seconds} segundos")
print(f"BPM: {bpm}")
print(f"Offset: {offset} segundos")
print(f"Primeros 10 medios beats (segundos): {result1['half_beat_times']}")
print(f"Tiempos reales con offset: {result1['real_times_with_offset']}")
print(f"Primeros 10 ticks: {result1['ticks']}")
"""