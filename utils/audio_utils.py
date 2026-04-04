import numpy as np

def procesar_data_musical(duracion, resolucion, offset, data):
    """
    Procesa data musical para crear un tensor interpolado de valores BPM y TS.
    Los valores máximos y mínimos se usan para el proceso de interpolación.
    
    Args:
        duracion (float): Duración en segundos
        resolucion (int): Resolución en ticks por negra
        offset (float): Offset en segundos
        data (list): Lista de diccionarios con información musical
    
    Returns:
        dict: Diccionario con valores procesados y tensores interpolados
    """
    
    # Valores por defecto más comunes
    BPM_DEFAULT = 120.0  # BPM más común en música popular
    TS_DEFAULT = 4       # Time signature 4/4 más común
    
    # Separar TS y BPM de la data
    ts_values = []
    ts_positions = []
    bpm_values = []
    posiciones_bpm = []
    
    for item in data:
        if item['type'] == 'TS':
            ts_values.append(int(item['data'][0]))
            ts_positions.append(item['position'])
        elif item['type'] == 'B':
            bpm_values.append(float(item['data'][0]) / 1000)  # Convertir a BPM real
            posiciones_bpm.append(item['position'])
    
    # Verificar si no existen valores y usar por defecto
    if not bpm_values:
        print("No se encontraron valores BPM, usando valor por defecto: 120 BPM")
        bpm_values = [BPM_DEFAULT]
        posiciones_bpm = [0]  # Posición inicial
    
    if not ts_values:
        print("No se encontraron valores TS, usando valor por defecto: 4/4")
        ts_values = [TS_DEFAULT]
        ts_positions = [0]  # Posición inicial
    
    # Calcular valores máximo y mínimo para BPM
    max_bpm_data = max(bpm_values)
    min_bpm_data = min(bpm_values)
    
    # Calcular valores máximo y mínimo para TS
    max_ts_data = max(ts_values)
    min_ts_data = min(ts_values)
    
    # Calcular ticks máximos usando el rango completo de BPM para mejor estimación
    bpm_promedio = np.mean(bpm_values)
    max_ticks = (duracion - offset) * resolucion * (bpm_promedio / 60)
    max_ticks_int = int(np.ceil(max_ticks))
    print(bpm_promedio)
    # Verificar si el cálculo con max_bpm_data da más ticks
    max_ticks_alt = (duracion - offset) * resolucion * (max_bpm_data / 60)
    print(max_bpm_data)
    if max_ticks_alt > max_ticks:
        max_ticks_int = int(np.ceil(max_ticks_alt))
        
    # Crear tensor de posiciones para interpolación
    tick_positions = np.arange(max_ticks_int)
    
    # Interpolar valores BPM
    if len(bpm_values) == 1:
        interpolated_bpm = np.full(max_ticks_int, bpm_values[0])
    else:
        # Extender las posiciones BPM si es necesario
        extended_bpm_positions = posiciones_bpm.copy()
        extended_bpm = bpm_values.copy()
        
        # Asegurar que el final del tensor tenga un valor definido
        if extended_bpm_positions[-1] < max_ticks_int - 1:
            extended_bpm_positions.append(max_ticks_int - 1)
            extended_bpm.append(bpm_values[-1])  # Mantener el último valor
        
        # Interpolar y aplicar límites basados en los extremos
        interpolated_bpm_raw = np.interp(tick_positions, extended_bpm_positions, extended_bpm)
        interpolated_bpm = np.clip(interpolated_bpm_raw, min_bpm_data, max_bpm_data)
    
    # Interpolar valores TS
    if len(ts_values) == 1:
        interpolated_ts = np.full(max_ticks_int, ts_values[0])
    else:
        # Extender las posiciones TS si es necesario
        extended_ts_positions = ts_positions.copy()
        extended_ts = ts_values.copy()
        
        # Asegurar que el final del tensor tenga un valor definido
        if extended_ts_positions[-1] < max_ticks_int - 1:
            extended_ts_positions.append(max_ticks_int - 1)
            extended_ts.append(ts_values[-1])  # Mantener el último valor
        
        # Para TS usamos interpolación por pasos (nearest) ya que son valores discretos
        interpolated_ts_raw = np.interp(tick_positions, extended_ts_positions, extended_ts)
        # Redondear a enteros y aplicar límites
        interpolated_ts = np.clip(np.round(interpolated_ts_raw).astype(int), min_ts_data, max_ts_data)
    
    return {
        #'ts_values': ts_values,
        #'ts_positions': ts_positions,
        #'bpm_values': bpm_values,
        #'posiciones_bmp': posiciones_bpm,
        'max_ticks': max_ticks_int,
        'tensor_bpm': interpolated_bpm,
        'tensor_ts': interpolated_ts,
        'tensor_shape': interpolated_bpm.shape,
        'used_defaults': {
            'bpm': len([item for item in data if item['type'] == 'B']) == 0,
            'ts': len([item for item in data if item['type'] == 'TS']) == 0
        }
    }