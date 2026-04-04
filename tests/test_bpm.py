import os
import librosa
from deeprhythm import DeepRhythmPredictor
from data.chart_loader import CloneHeroChartParser
from models.tokenizer import CloneHeroTokenizer
from pydub import AudioSegment
from typing import List, Dict, Any, Optional, Tuple, Union
from utils.time_utils import song_duration_to_ticks_complete, ticks_to_seconds_with_offset
from itertools import product
import base64
import json
import numpy as np
import torch
from models.demucs import DemucsAudioSeparator
from datasets import Dataset
from tqdm import tqdm
from models.mert import MERT

max_duration_minutes = 120
max_beats = 512
max_beatshifts = 32

tokenizer = CloneHeroTokenizer()
mert = MERT()


def get_time_signature(sync_track: List[Dict[str, Any]]) -> int:
    """
    Obtiene la signatura de tiempo (Time Signature) del sync track.
    
    Args:
        sync_track: Lista de elementos del sync track
        
    Returns:
        Signatura de tiempo como entero. Si hay múltiples TS, retorna 4 por defecto.
        Si hay solo uno, retorna ese valor.
    """
    # Filtrar elementos de tipo "TS"
    ts_elements = [item for item in sync_track if item['type'] == 'TS']
    
    # Si no hay elementos TS, usar 4 por defecto
    if not ts_elements:
        return 4
    
    # Si hay más de un elemento TS, usar 4 por defecto
    if len(ts_elements) > 1:
        return 4
    
    # Si hay exactamente uno, usar ese valor
    ts_value = int(ts_elements[0]['data'][0])
    return ts_value

def load_chart_and_audio_from_path(path: str) -> Dict[str, Any]:
    """
    Carga el contenido del archivo chart y los streams de audio desde un path dado.
    
    Args:
        path: Ruta a la carpeta que contiene los archivos
        
    Returns:
        Diccionario con contenido del chart y rutas de streams
    """
    chart_path = os.path.join(path, 'notes.chart')
    song_ini_path = os.path.join(path, 'song.ini')
    streams = {}
    chart_content = None
    song_ini_content = None

    # Leer archivo chart
    if os.path.exists(chart_path):
        with open(chart_path, 'r', encoding='utf-8') as f:
            chart_content = f.read()

    # Leer archivo song.ini
    if os.path.exists(song_ini_path):
        with open(song_ini_path, 'r', encoding='utf-8') as f:
            song_ini_content = f.read()

    # Buscar archivos de audio
    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith('.ogg') or file.endswith('.mp3') or file.endswith('.wav'):
                streams[file] = os.path.join(path, file)

    return {
        'chart_content': chart_content,
        'song_ini_content': song_ini_content,
        'streams': streams
    }

def map_streams_to_metadata(streams: Dict[str, str]) -> Dict[str, str]:
    """
    Mapea archivos de audio a las claves esperadas por el parser.
    
    Args:
        streams: Diccionario con nombres de archivo y rutas
        
    Returns:
        Diccionario con streams mapeados a claves estándar
    """
    metadata = {}
    
    for filename, filepath in streams.items():
        filename_lower = filename.lower()
        
        if 'guitar' in filename_lower:
            metadata['GuitarStream'] = filepath
        elif 'bass' in filename_lower:
            metadata['BassStream'] = filepath
        elif 'vocals' in filename_lower or 'vocal' in filename_lower:
            metadata['VocalStream'] = filepath
        elif 'drum' in filename_lower:
            metadata['DrumStream'] = filepath
        elif 'keys' in filename_lower or 'keyboard' in filename_lower:
            metadata['KeysStream'] = filepath
        elif 'song' in filename_lower or 'track' in filename_lower:
            metadata['SongStream'] = filepath
    
    return metadata

def count_b_elements(sync_track: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cuenta y filtra elementos de tipo 'B' del sync track.
    """
    return [item for item in sync_track if item['type'] == 'B']

def calculate_bpm_from_single_b_element(b_element: Dict[str, Any]) -> float:
    """
    Calcula BPM a partir de un único elemento tipo 'B'.
    """
    bpm_value = int(b_element['data'][0])
    return bpm_value / 1000

def calculate_bpm_from_song_stream(song_stream_path: str) -> float:
    """
    Calcula BPM usando DeepRhythm sobre el archivo SongStream.
    """
    model = DeepRhythmPredictor()
    bpm, confidence = model.predict(song_stream_path, include_confidence=True)
    return bpm

def get_available_streams(song_info: Dict[str, Any]) -> List[str]:
    """
    Obtiene lista de streams de audio disponibles y válidos.
    """
    stream_keys = ['GuitarStream', 'BassStream', 'VocalStream', 'DrumStream', 'KeysStream']
    available_streams = []
    
    #  GuitarStream = "guitar.ogg" BassStream = "bass.ogg" VocalStream = "vocals.ogg"
    for key in stream_keys:
        if key in song_info:
            stream_path = song_info[key]
            if os.path.exists(stream_path):
                available_streams.append(stream_path)
    
    return available_streams

def load_audio_file(file_path: str) -> Optional[AudioSegment]:
    """
    Carga un archivo de audio usando pydub.
    """
    try:
        if file_path.endswith('.ogg'):
            return AudioSegment.from_ogg(file_path)
        elif file_path.endswith('.mp3'):
            return AudioSegment.from_mp3(file_path)
        elif file_path.endswith('.wav'):
            return AudioSegment.from_wav(file_path)
        else:
            print(f"Formato no soportado: {file_path}")
            return None
    except Exception as e:
        print(f"Error al cargar {file_path}: {e}")
        return None

def combine_audio_streams(stream_paths: List[str]) -> AudioSegment:
    """
    Combina múltiples archivos de audio en uno solo.
    """
    combined_audio = None
    
    for stream_path in stream_paths:
        audio_segment = load_audio_file(stream_path)
        
        if audio_segment is not None:
            if combined_audio is None:
                combined_audio = audio_segment
            else:
                combined_audio = combined_audio.overlay(audio_segment)
    
    if combined_audio is None:
        raise ValueError("No se pudo cargar ningún archivo de audio")
    
    return combined_audio

def calculate_bpm_from_combined_audio(combined_audio: AudioSegment) -> float:
    """
    Calcula BPM a partir de audio combinado usando DeepRhythm y librosa.
    """
    temp_file = "temp_combined_audio.wav"
    
    try:
        combined_audio.export(temp_file, format="wav")
        audio, sr = librosa.load(temp_file)
        model = DeepRhythmPredictor()
        bpm = model.predict_from_audio(audio, sr)
        return bpm
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def calculate_bpm_from_multiple_streams(song_info: Dict[str, Any]) -> float:
    """
    Calcula BPM combinando múltiples streams de audio.
    """
    available_streams = get_available_streams(song_info)
    
    if not available_streams:
        raise ValueError("No se encontraron archivos de audio válidos")
    
    combined_audio = combine_audio_streams(available_streams)
    return calculate_bpm_from_combined_audio(combined_audio)

def calculate_bpm_from_folder(parser, streams) -> tuple:
    """
    Calcula BPM y regresa bpm, ts, duracion, resolucion y offset.
    """
    sync_track = parser.get_sync_track()
    b_elements = count_b_elements(sync_track)
    time_signature = get_time_signature(sync_track)
    song_info = parser.get_song_metadata()
    resolution = song_info.get("Resolution", 192)
    offset = song_info.get("Offset", 0.0)
    # Buscar duración en streams
    duration = None
    for stream_path in streams.values():
        try:
            audio = AudioSegment.from_file(stream_path)
            duration = audio.duration_seconds
            break
        except Exception:
            continue
    if duration is None:
        duration = 1.0
    # Caso 1: Solo un elemento tipo "B"
    if len(b_elements) == 1:
        bpm = calculate_bpm_from_single_b_element(b_elements[0])
        return bpm, time_signature, duration, resolution, offset
    # Caso 2: Múltiples elementos "B" - usar análisis de audio
    original_metadata = parser.get_song_metadata()
    stream_metadata = map_streams_to_metadata(streams)
    combined_metadata = {**original_metadata, **stream_metadata}
    if 'SongStream' in combined_metadata:
        bpm = calculate_bpm_from_song_stream(combined_metadata['SongStream'])
        return bpm, time_signature, duration, resolution, offset
    bpm = calculate_bpm_from_multiple_streams(combined_metadata)
    return bpm, time_signature, duration, resolution, offset

def encode_metadata_base64(metadata):
    json_str = json.dumps(metadata)
    b64_str = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    return b64_str

def instrument_to_token(instrument: str) -> str:
    """
    Convierte el nombre del instrumento a un token específico.
    
    Args:
        instrument: Nombre del instrumento (ej. 'Guitar', 'Bass', etc.)

    Returns:
        Token correspondiente al instrumento.
    """
    instrument_tokens = {
        'Single': '<Guitar>',
        'DoubleRhythm': '<Guitar>',
        'GuitarCoop': '<Guitar>',
        'DoubleBass': '<Bass>',
        'DoubleGuitar': '<Guitar>',
        'Drum': '<Drums>',
    }
    #<Guitar>', '<Bass>', '<Drums>
    return instrument_tokens.get(instrument, '<UNK>')


def ticks_to_minutes_beats_beatshift(ticks: int, bpm: float, resolution: int = 192, offset=0) -> Tuple[int, int, int]:
    """
    Convierte ticks a minutos, beats y beatshift.
    
    Args:
        ticks: Número de ticks
        bpm: Beats por minuto
        resolution: Resolución MIDI (default 480)
        
    Returns:
        Tuple con minutos, beats y beatshift
    """
    
    time = ticks_to_seconds_with_offset(ticks, bpm, resolution=resolution, offset=offset)
    
    minutes = int(time // 60)
    seconds = time % 60
    beats = int(seconds * (bpm / 60))
    beatshift = int(((ticks*max_beatshifts) /resolution ))
    return minutes, beats, beatshift


def note_to_pretokenizer(note: Dict[str, Any], notes_range: Tuple, bpm: float, resolution: int, offset=0) -> List[Any]:
    """
    Prepara una nota para la tokenización.
    Args:
        note: Diccionario con información de la nota (ej. posición, tipo, botón, duración)
    Returns:
        Lista con los valores necesarios para la tokenización.
    """
    note_types = {
        1: 'normal',
        2: 'special',
    }

    beat_shift_ticks = (notes_range[-1] - notes_range[0]) // max_beatshifts # 6 si resolucion es 192 y max_beatshifts = 32_
    init = abs(note.get('position', notes_range[0]) - notes_range[0])
    init_beat_shift = round(init / beat_shift_ticks)

    note_type = note_types.get(note.get('type', 1), 'normal') 

    pitch = note.get('button', 0)  # Asumiendo que 'button' es el pitch

    duration_ticks = note.get('duration', 0)  # La duración está en ticks
    minutes, beats, beatshift = ticks_to_minutes_beats_beatshift(duration_ticks, bpm, resolution, offset=offset)
    #(init_beat_shift, note_type, pitch, duration_minutes, duration_beats, duration_beatshift)
    return init_beat_shift, note_type, pitch, minutes, beats, beatshift

def tokenize_notes(notes : List[Dict[str, Any]], instrument: str, difficulty: str, tokenizer: Any, bpm:float, resolution: int, offset: int, notes_range: tuple[int, int]) -> List[str]:
    """
    Tokeniza las notas de un chart.
    
    Returns:
        Lista de tokens representando las notas.
    """
    # All notes are in the range of notes_range
    # All notes are ordered by position
    # beat_sequence (init_beat_shift, note_type, pitch, duration_minutes, duration_beats, duration_beatshift)
    beat_sequence = []
    for note in notes:
        sequence = note_to_pretokenizer(note, notes_range, bpm, resolution, offset)
        beat_sequence.append(sequence)

    instrument = instrument_to_token(instrument)

    tokens = tokenizer.encode_complete_chart(
        instrument=instrument,
        difficulty=f'<{difficulty}>',
        beat_sequence=beat_sequence,
    )
    return tokens

def get_audio_info(segments: List[np.ndarray], hop_length: int = 1024, sample_rate: int = 22050, is_padded: bool = False):
    """
    Proporciona información útil sobre los segmentos extraídos.
    
    Args:
        segments: Lista de segmentos del log-mel spectrogram
        hop_length: Salto entre frames usado en la extracción
        sample_rate: Frecuencia de muestreo
        is_padded: Si los segmentos tienen padding aplicado
    
    Returns:
        Diccionario con información de cada segmento
    """
    info = []
    
    for i, segment in enumerate(segments):
        n_mels, n_frames = segment.shape
        duration_sec = n_frames * hop_length / sample_rate
        
        info.append({
            'segment_id': i,
            'shape': segment.shape,
            'duration_seconds': round(duration_sec, 3),
            'n_mels': n_mels,
            'n_frames': n_frames,
            'is_padded': is_padded
        })
    
    return info


def pad_segments(log_S, sr, time_ranges, hop_length, pad_to_max_length) -> List[np.ndarray]:
    # Si no hay rangos de tiempo, devolver el spectrogram completo
    if time_ranges is None or len(time_ranges) < 2:
        return [log_S]
    
    # Segmentar según los rangos de tiempo
    segments = []
    max_frames = 0
    temp_segments = []
    
    for i in range(len(time_ranges) - 1):
        start_sec = time_ranges[i]
        end_sec = time_ranges[i + 1]
        
        # Convertir segundos a índices de frames
        start_frame = int(np.floor(start_sec * sr / hop_length))
        end_frame = int(np.floor(end_sec * sr / hop_length))
        
        # Asegurar que los índices estén dentro del rango válido
        start_frame = max(0, start_frame)
        end_frame = min(log_S.shape[1], end_frame)
        
        # Extraer segmento
        if start_frame < end_frame:
            segment = log_S[:, start_frame:end_frame]
            temp_segments.append(segment)
            max_frames = max(max_frames, segment.shape[1])
    
    # Si se requiere padding, hacer padding con 0s
    if pad_to_max_length:
        for seg in temp_segments:
            n_mels, n_frames = seg.shape
            if n_frames < max_frames:
                # Hacer padding a la derecha con columnas de 0
                pad_width = max_frames - n_frames
                pad_array = np.zeros((n_mels, pad_width), dtype=seg.dtype)
                padded_seg = np.concatenate((seg, pad_array), axis=1)
                segments.append(padded_seg)
            else:
                segments.append(seg)
    else:
        segments = temp_segments
    
    return segments


def extract_logmel_spectrogram_segments(
    audio_input: Union[str, np.ndarray], 
    sample_rate: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    f_min: float = 10.0,
    n_mels: int = 512,
    time_ranges: Optional[List[float]] = None,
    pad_to_max_length: bool = False
) -> List[np.ndarray]:
    """
    Extrae un log-mel spectrogram de audio y lo segmenta según rangos de tiempo.
    Opcionalmente hace padding con 0s para que todos los segmentos tengan la misma cantidad de frames.
    
    Args:
        audio_input: Ruta del archivo de audio (str) o array de audio (np.ndarray)
        sample_rate: Frecuencia de muestreo (22050 Hz por defecto)
        n_fft: Tamaño de la ventana FFT (4096 por defecto)
        hop_length: Salto entre frames (1024 por defecto)
        f_min: Frecuencia mínima en Hz (10.0 por defecto)
        n_mels: Número de bandas mel (512 por defecto)
        time_ranges: Lista de tiempos en segundos para segmentar [0, 0.3, 0.6, ...]
        pad_to_max_length: Si es True, se hace padding con 0s para que todos los segmentos tengan la misma longitud
    
    Returns:
        Lista de arrays numpy con los segmentos del log-mel spectrogram
        Cada segmento tiene forma (n_mels, max_frames) si pad_to_max_length=True
    """
    
    # Cargar audio según el tipo de entrada
    if isinstance(audio_input, str):
        # Cargar desde archivo
        y, sr = librosa.load(audio_input, sr=sample_rate)
    else:
        # Usar array directamente
        y = audio_input
        sr = sample_rate
    
    # Calcular mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=f_min,
        n_mels=n_mels
    )
    
    # Convertir a escala logarítmica (dB)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    return pad_segments(log_S, sr, time_ranges, hop_length, pad_to_max_length)


#song_info, streams, inst, tick_ranges
def calculate_logmel_spectrogram(parser: Any, streams: List[str], inst: str, tick_ranges: List[int], separator: Any, combined_metadata: Any, mert: Any) -> List[str]:
    """
    Calcula el espectrograma log-mel de los streams de audio.
    
    Args:
        streams: Diccionario con nombres de archivo y rutas
    Returns:
        Lista de espectrogramas log-mel para cada stream
    """

    instruments_map = {
        'Single': 'GuitarStream',
        'DoubleRhythm': 'GuitarStream',
        'DoubleBass': 'BassStream',
        'Vocals': 'VocalStream',
        'Drums': 'DrumsStream',
    }

    #'guitar_other', 'bass', 'drums'
    instrument_sep_map = {
        'Single': 'guitar_other',
        'DoubleRhythm': 'guitar_other',
        'DoubleBass': 'bass',
        #'Vocals': 'VocalStream',
        'Drums': 'drums',

    }

    if 'SongStream' in combined_metadata:
        sr = separator.audio_processor.target_sample_rate
        stem = instrument_sep_map.get(inst, 'guitar_other')
        logmel = separator.calculate_stem_logmel(stem, n_mels=512, n_fft=4096, hop_length=1024)
        audio = separator.get_stem(stem, as_tensor=True)
        embeddings = mert.from_tensor(audio, sample_rate=sr)
        segments = pad_segments(logmel, sr, tick_ranges, 1024, pad_to_max_length= True)
    else:
        available_streams = get_available_streams(combined_metadata)
        if not available_streams:
            raise ValueError("No se encontraron archivos de audio válidos")
        audio_path = combined_metadata[instruments_map.get(inst)]
        embeddings = mert(audio_path)
        segments = extract_logmel_spectrogram_segments(
            audio_input=audio_path,
            time_ranges=tick_ranges,
            pad_to_max_length=True
        )

    return np.asarray(segments), embeddings


def convert_single_item_to_hf_format(data_dict):
    """
    Convierte un diccionario individual a formato compatible con Hugging Face Dataset
    """
    converted_data = {}
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            # Convertir tensores de PyTorch a listas de Python
            converted_data[key] = value.numpy().tolist()
        elif isinstance(value, tuple):
            # Convertir tuplas a listas para compatibilidad
            converted_data[key] = list(value)
        else:
            # Mantener otros tipos de datos como están
            converted_data[key] = value
    
    return converted_data


def create_hf_dataset_from_folder(folder_path: str, process_file_func) -> Dataset:
    """
    Procesa todos los archivos de una carpeta y crea un Dataset de Hugging Face
    
    Args:
        folder_path: Ruta de la carpeta con los archivos
        process_file_func: Función que procesa cada archivo y devuelve tu diccionario de datos
    
    Returns:
        Dataset de Hugging Face
    """
    all_data = []
    
    with open("chart_analysis_results.json", 'r', encoding='utf-8') as f:
        chart_analysis_results = json.load(f)

    folders_with_charts = chart_analysis_results['folders_with_chart']
    # Procesar cada archivo en la carpeta
    #[os.path.join(folder_path, folder) for folder in os.listdir(folder_path)]

    for path in tqdm(folders_with_charts):
        if not os.path.isdir(path):
            print(f"Error: La carpeta '{path}' no existe")
            continue
        # Procesar el archivo (aquí usarías tu función existente)
        data = process_file_func(path)
        if data is None:
            with open("error_log.txt", 'a', encoding='utf-8') as error_file:
                error_file.write(f"Error: No se pudo procesar el archivo en '{path}'\n")
            print(f"Error: No se pudo procesar el archivo en '{path}'")
            continue    
        for _data in data:
            # Convertir el diccionario a formato compatible
            converted_data = convert_single_item_to_hf_format(_data)
            all_data.append(converted_data)
    
    # Reorganizar datos para el formato de Dataset
    dataset_dict = {}
    
    # Obtener todas las claves del primer elemento
    if all_data:
        for key in all_data[0].keys():
            dataset_dict[key] = [item[key] for item in all_data]
    
    # Crear Dataset de Hugging Face
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def process_song(folder_path):
    try:
        if not os.path.exists(folder_path):
            print(f"Error: La carpeta '{folder_path}' no existe")
            return
        data = load_chart_and_audio_from_path(folder_path)
        chart_content = data['chart_content']
        streams = data['streams']
        if chart_content is None:
            raise FileNotFoundError(f"No se encontró el archivo notes.chart en {folder_path}")
        
        parser = CloneHeroChartParser(chart_content)
        bpm, ts, duration, resolution, offset = calculate_bpm_from_folder(parser, streams)
        song_info = parser.get_song_metadata()
        print("Metadata:", song_info)
        #print("Metadata base64:", encode_metadata_base64(song_info))
        print(f"BPM calculado para '{folder_path}': {bpm}, Time Signature: {ts}, Duración: {duration}, Resolución: {resolution}, Offset: {offset}")
        tick_data = song_duration_to_ticks_complete(bpm, duration, resolution, offset)
        #print("Tick data:", tick_data)
        instruments = parser.get_available_instruments()
        print("Instrumentos:", instruments)
        difficulties = parser.get_available_difficulties()
        print("Dificultades:", difficulties)
        tick_ranges = tick_data['ticks']
        #print("Tick ranges:", tick_ranges)

        all_song_data = []

        for inst, diff in list(product(instruments, difficulties)):
            notes = parser.get_instrument_track_numeric(inst, diff)
            print(f"Notas para {inst} - {diff} (total: {len(notes)}):")
            song_tokens = []
            for i in range(len(tick_ranges)-1):
                start_tick = tick_ranges[i]
                end_tick = tick_ranges[i+1]
                notes_in_range = [n for n in notes if start_tick <= n['position'] < end_tick]
                ind_song_token = tokenize_notes(notes_in_range, inst, diff, tokenizer, bpm, resolution, offset, (start_tick, end_tick))
                song_tokens.append(ind_song_token)

            original_metadata = parser.get_song_metadata()
            stream_metadata = map_streams_to_metadata(streams)
            combined_metadata = {**original_metadata, **stream_metadata}
            separator = None
            if 'SongStream' in combined_metadata:
                separator = DemucsAudioSeparator(combined_metadata["SongStream"], model_name="htdemucs")

            logmel_spectrogram, song_embeddings = calculate_logmel_spectrogram(parser, streams, inst, tick_data["beat_times"], separator, combined_metadata, mert)

            data = {
                'logmel_spectrogram': torch.Tensor(logmel_spectrogram),
                'metadata': encode_metadata_base64(song_info),
                'instrument': inst,
                'difficulty': diff,
                'bpm': bpm,
                'time_signature': ts,
                'resolution': resolution,
                'offset': offset,
                'song_tokens': song_tokens,
                'tick_ranges': tick_ranges,
                'beat_times': tick_data["beat_times"],
                'song_embeddings': song_embeddings,
            }
            all_song_data.append(data)
        return all_song_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error al calcular BPM: {e}")

if __name__ == "__main__":
    #folder_path = "Grupo Marca Registrada - El Rescate"
    #folder_path = "El Precio de la Soledad"
    dataset = create_hf_dataset_from_folder(
        folder_path="G:\\Songs",
        #folder_path="./test_dataset",
        process_file_func=process_song
    )
    dataset.save_to_disk("clone_hero_dataset")
    import pdb;pdb.set_trace()
    

    # Cargar el dataset posteriormente
    #from datasets import load_from_disk
    #loaded_dataset = load_from_disk("clone_hero_dataset")

    

