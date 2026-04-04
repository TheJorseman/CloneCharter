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
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from models.mert import MERT
from datetime import datetime
import tempfile
import gc

max_duration_minutes = 120
max_beats = 512
max_beatshifts = 32

tokenizer = CloneHeroTokenizer()
mert = MERT()


def save_checkpoint_metadata(processed_files: List[str], current_index: int, total_files: int, 
                           total_samples: int, checkpoint_dir: str):
    """
    Guarda metadata del checkpoint en formato JSON
    
    Args:
        processed_files: Lista de archivos ya procesados
        current_index: Índice actual en la lista de archivos
        total_files: Total de archivos a procesar
        total_samples: Total de muestras procesadas
        checkpoint_dir: Directorio donde guardar la metadata
    """
    metadata = {
        'processed_files': processed_files,
        'current_index': current_index,
        'total_files': total_files,
        'total_samples': total_samples,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    }
    
    # Crear directorio si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Guardar metadata
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metadata guardada: {current_index}/{total_files} archivos procesados, {total_samples} muestras totales")


def load_checkpoint_metadata(checkpoint_dir: str) -> Optional[Dict]:
    """
    Carga metadata del checkpoint desde JSON
    
    Args:
        checkpoint_dir: Directorio donde buscar la metadata
        
    Returns:
        Diccionario con los datos del checkpoint o None si no existe
    """
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"📂 Checkpoint encontrado:")
        print(f"   - Archivos procesados: {len(metadata['processed_files'])}")
        print(f"   - Progreso: {metadata['current_index']}/{metadata['total_files']}")
        print(f"   - Muestras totales: {metadata['total_samples']}")
        print(f"   - Fecha: {metadata['timestamp']}")
        
        return metadata
    
    except Exception as e:
        print(f"❌ Error al cargar metadata del checkpoint: {e}")
        return None


def load_existing_dataset(dataset_path: str) -> Optional[Dataset]:
    """
    Carga un dataset existente desde disco
    
    Args:
        dataset_path: Ruta del dataset a cargar
        
    Returns:
        Dataset cargado o None si no existe
    """
    if not os.path.exists(dataset_path):
        return None
    
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"📂 Dataset existente cargado: {len(dataset)} muestras")
        return dataset
    except Exception as e:
        print(f"❌ Error al cargar dataset existente: {e}")
        return None


def save_dataset_incremental(dataset: Dataset, dataset_path: str):
    """
    Guarda el dataset de forma incremental
    
    Args:
        dataset: Dataset a guardar
        dataset_path: Ruta donde guardar el dataset
    """
    try:
        dataset.save_to_disk(dataset_path, max_shard_size="1GB")
        print(f"💾 Dataset guardado: {len(dataset)} muestras")
    except Exception as e:
        print(f"❌ Error al guardar dataset: {e}")
        raise


def clean_checkpoint_files(checkpoint_dir: str):
    """
    Elimina archivos de checkpoint
    
    Args:
        checkpoint_dir: Directorio de checkpoints a limpiar
    """
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        print(f"🗑️ Metadata de checkpoint eliminada")


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

def calculate_bpm_from_folder(parser, streams, song_stream_path) -> tuple:
    """
    Calcula BPM y regresa bpm, ts, duracion, resolucion y offset.
    """
    sync_track = parser.get_sync_track()
    b_elements = count_b_elements(sync_track)
    time_signature = get_time_signature(sync_track)
    song_info = parser.get_song_metadata()
    resolution = song_info.get("Resolution", 192)
    offset = song_info.get("Offset", 0.0)
    
    try:
        audio = AudioSegment.from_file(song_stream_path)
        duration = audio.duration_seconds
    except Exception:
        duration = 1.0
    # Caso 1: Solo un elemento tipo "B"
    if len(b_elements) == 1:
        bpm = calculate_bpm_from_single_b_element(b_elements[0])
        return bpm, time_signature, duration, resolution, offset
    bpm = calculate_bpm_from_song_stream(song_stream_path)
    return bpm, time_signature, duration, resolution, offset


def encode_metadata_base64(metadata):
    json_str = json.dumps(metadata)
    b64_str = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    return b64_str


def instrument_to_token(instrument: str) -> str:
    """
    Convierte el nombre del instrumento a un token específico.
    """
    instrument_tokens = {
        'Single': '<Guitar>',
        'DoubleRhythm': '<Guitar>',
        'GuitarCoop': '<Guitar>',
        'DoubleBass': '<Bass>',
        'DoubleGuitar': '<Guitar>',
        'Drum': '<Drums>',
    }
    return instrument_tokens.get(instrument, '<UNK>')


def ticks_to_minutes_beats_beatshift(ticks: int, bpm: float, resolution: int = 192, offset=0) -> Tuple[int, int, int]:
    """
    Convierte ticks a minutos, beats y beatshift.
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
    """
    note_types = {
        1: 'normal',
        2: 'special',
    }

    beat_shift_ticks = (notes_range[-1] - notes_range[0]) // max_beatshifts
    init = abs(note.get('position', notes_range[0]) - notes_range[0])
    init_beat_shift = round(init / beat_shift_ticks)

    note_type = note_types.get(note.get('type', 1), 'normal') 
    pitch = note.get('button', 0)

    duration_ticks = note.get('duration', 0)
    minutes, beats, beatshift = ticks_to_minutes_beats_beatshift(duration_ticks, bpm, resolution, offset=offset)
    
    return init_beat_shift, note_type, pitch, minutes, beats, beatshift


def tokenize_notes(notes : List[Dict[str, Any]], instrument: str, difficulty: str, tokenizer: Any, bpm:float, resolution: int, offset: int, notes_range: tuple[int, int]) -> List[str]:
    """
    Tokeniza las notas de un chart.
    """
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


def calculate_logmel_spectrogram(inst: str, tick_ranges: List[int], separator: Any, combined_metadata: Any) -> List[str]:
    """
    Calcula el espectrograma log-mel de los streams de audio.
    """
    instruments_map = {
        'Single': 'GuitarStream',
        'DoubleRhythm': 'GuitarStream',
        'DoubleBass': 'BassStream',
        'Vocals': 'VocalStream',
        'Drums': 'DrumsStream',
    }

    instrument_sep_map = {
        'Single': 'guitar_other',
        'DoubleRhythm': 'guitar_other',
        'DoubleBass': 'bass',
        'Drums': 'drums',
    }

    if 'SongStream' in combined_metadata:
        sr = separator.audio_processor.target_sample_rate
        stem = instrument_sep_map.get(inst, 'guitar_other')
        logmel = separator.calculate_stem_logmel(stem, n_mels=512, n_fft=4096, hop_length=1024)
        segments = pad_segments(logmel, sr, tick_ranges, 1024, pad_to_max_length= True)
    else:
        available_streams = get_available_streams(combined_metadata)
        if not available_streams:
            raise ValueError("No se encontraron archivos de audio válidos")
        audio_path = combined_metadata[instruments_map.get(inst)]
        segments = extract_logmel_spectrogram_segments(
            audio_input=audio_path,
            time_ranges=tick_ranges,
            pad_to_max_length=True
        )

    return np.asarray(segments)


def convert_single_item_to_hf_format(data_dict):
    """
    Convierte un diccionario individual a formato compatible con Hugging Face Dataset
    con tipos de datos consistentes
    """
    converted_data = {}
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            # Convertir a float16 para ahorrar espacio
            converted_data[key] = value.half().numpy().tolist()
        elif isinstance(value, tuple):
            # Convertir tuplas a listas
            converted_data[key] = list(value)
        elif key in ['bpm', 'offset']:  # **CAMBIO CLAVE**
            # Asegurar que bpm y offset sean siempre float64
            converted_data[key] = float(value)
        elif key in ['time_signature', 'resolution']:
            # Asegurar que estos sean siempre int64
            converted_data[key] = int(value)
        elif key == 'song_tokens' and isinstance(value, list):
            # Asegurar que tokens sean int64 consistentemente
            converted_data[key] = [
                [int(token) if isinstance(token, (int, float)) else token for token in sequence] 
                for sequence in value
            ]
        elif key in ['tick_ranges', 'beat_times'] and isinstance(value, list):
            # Asegurar que sean float32
            converted_data[key] = [float(x) for x in value]
        elif key == 'song_embeddings' and hasattr(value, 'tolist'):
            # Asegurar que embeddings sean float16
            if isinstance(value, torch.Tensor):
                converted_data[key] = value.half().numpy().tolist()
            else:
                converted_data[key] = [float(x) for x in value.flatten()]
        else:
            # Mantener otros tipos como están
            converted_data[key] = value
    
    return converted_data



def save_dataset_with_concatenation_efficient(new_batch_dataset: Dataset, dataset_path: str):
    """
    Concatena eficientemente sin mantener datasets grandes en memoria
    """
    temp_path = dataset_path + "_temp"
    try:
        # Si existe dataset anterior, concatenar
        if os.path.exists(dataset_path):
            # Cargar solo para concatenar, no mantener en memoria
            existing_dataset = Dataset.load_from_disk(dataset_path)
            combined_dataset = concatenate_datasets([existing_dataset, new_batch_dataset])
            # Guardar en ubicación temporal
            combined_dataset.save_to_disk(temp_path)
            # Liberar memoria inmediatamente
            del existing_dataset
            del combined_dataset
            # Reemplazar el original
            import shutil
            shutil.rmtree(dataset_path)
            shutil.move(temp_path, dataset_path)
        else:
            # Primera vez: guardar directamente
            new_batch_dataset.save_to_disk(dataset_path)
        print(f"💾 Dataset actualizado exitosamente")
        
    except Exception as e:
        # Limpiar archivos temporales en caso de error
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        raise e

def create_hf_dataset_incremental(
    folder_path: str, 
    process_file_func,
    batch_size: int = 5,
    save_interval: int = 10,
    checkpoint_dir: str = "./checkpoints",
    dataset_path: str = "./clone_hero_dataset",
    resume_from_checkpoint: bool = True
) -> Dataset:
    """
    Procesa archivos y crea un Dataset de Hugging Face de forma incremental
    
    Args:
        folder_path: Ruta de la carpeta con los archivos
        process_file_func: Función que procesa cada archivo
        batch_size: Tamaño del lote para procesar
        save_interval: Cada cuántos archivos guardar el dataset
        checkpoint_dir: Directorio para metadata de checkpoint
        dataset_path: Ruta donde guardar el dataset
        resume_from_checkpoint: Si reanudar desde checkpoint existente
    
    Returns:
        Dataset de Hugging Face
    """
    
    # Cargar lista de archivos a procesar
    with open("chart_analysis_results.json", 'r', encoding='utf-8') as f:
        chart_analysis_results = json.load(f)
    
    folders_with_charts = chart_analysis_results['folders_with_chart']
    total_files = len(folders_with_charts)
    
    # Variables para tracking
    processed_files = []
    start_index = 0
    total_samples = 0
    main_dataset = None
    
    # Intentar cargar checkpoint y dataset existente
    if resume_from_checkpoint:
        # Cargar metadata
        checkpoint_metadata = load_checkpoint_metadata(checkpoint_dir)
        if checkpoint_metadata is not None:
            processed_files = checkpoint_metadata['processed_files']
            start_index = checkpoint_metadata['current_index']
            total_samples = checkpoint_metadata['total_samples']
            
            print(f"🔄 Reanudando desde checkpoint...")
            print(f"   - Archivos ya procesados: {len(processed_files)}")
            print(f"   - Continuando desde índice: {start_index}")
            print(f"   - Muestras totales: {total_samples}")
        
        # Cargar dataset existente
        main_dataset = load_existing_dataset(dataset_path)
    
    # Procesar archivos restantes
    try:
        batch_data = []
        
        for i in tqdm(range(start_index, total_files), desc="Procesando archivos"):
            path = folders_with_charts[i]
            
            # Verificar si ya fue procesado
            if path in processed_files:
                continue
            
            if not os.path.isdir(path):
                print(f"❌ Error: La carpeta '{path}' no existe")
                processed_files.append(path)
                continue
            
            # Procesar el archivo
            try:
                data = process_file_func(path)
                if data is None:
                    with open("error_log.txt", 'a', encoding='utf-8') as error_file:
                        error_file.write(f"Error: No se pudo procesar el archivo en '{path}'\n")
                    print(f"❌ Error: No se pudo procesar el archivo en '{path}'")
                    processed_files.append(path)
                    continue
                
                # Convertir y agregar a lote
                for _data in data:
                    converted_data = convert_single_item_to_hf_format(_data)
                    batch_data.append(converted_data)
                
                processed_files.append(path)
                
                # Procesar lote cuando alcance el tamaño deseado o sea el último archivo
                if len(batch_data) >= batch_size or i == total_files - 1:
                    if batch_data:
                        # Crear dataset del lote
                        batch_dict = {}
                        for key in batch_data[0].keys():
                            batch_dict[key] = [item[key] for item in batch_data]
                        
                        batch_dataset = Dataset.from_dict(batch_dict)
                        
                        save_dataset_with_concatenation_efficient(batch_dataset, dataset_path)

                        # Actualizar contadores
                        total_samples += len(batch_data)
                        print(f"📊 Lote procesado: +{len(batch_data)} muestras, Total: {total_samples}")
                        del batch_dataset
                        batch_data = []
                        
                        # Guardar metadata
                        save_checkpoint_metadata(processed_files, i + 1, total_files, 
                                            total_samples, checkpoint_dir)
                        # Forzar limpieza de memoria
                        gc.collect()                        
                        # Concatenar con dataset principal
                        #if main_dataset is None:
                        #    main_dataset = batch_dataset
                        #else:
                        #    main_dataset = concatenate_datasets([main_dataset, batch_dataset])
                        
                        #total_samples = len(main_dataset)
                        #print(f"📊 Lote procesado: +{len(batch_data)} muestras, Total: {total_samples}")
                        #batch_data = []
                # Guardar dataset y metadata cada save_interval archivos
                #if (i + 1) % save_interval == 0 or i == total_files - 1:
                #if main_dataset is not None:
                #    save_dataset_incremental(main_dataset, dataset_path)
                #    save_checkpoint_metadata(processed_files, i + 1, total_files, 
                #                            total_samples, checkpoint_dir)
                        
            except Exception as e:
                # Log del error y continuar
                error_msg = f"Error procesando '{path}': {str(e)}\n"
                with open("error_log.txt", 'a', encoding='utf-8') as error_file:
                    error_file.write(error_msg)
                print(f"❌ {error_msg.strip()}")
                processed_files.append(path)
                continue
    
    except KeyboardInterrupt:
        print("\n⚠️ Procesamiento interrumpido por el usuario")
        # Guardar estado actual
        if main_dataset is not None:
            save_dataset_incremental(main_dataset, dataset_path + "_emergency")
            emergency_metadata_dir = checkpoint_dir + "_emergency"
            save_checkpoint_metadata(processed_files, len(processed_files), total_files, 
                                   len(main_dataset), emergency_metadata_dir)
            print(f"💾 Estado de emergencia guardado")
        raise
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        # Guardar estado actual
        if main_dataset is not None:
            save_dataset_incremental(main_dataset, dataset_path + "_emergency")
            emergency_metadata_dir = checkpoint_dir + "_emergency"
            save_checkpoint_metadata(processed_files, len(processed_files), total_files, 
                                   len(main_dataset), emergency_metadata_dir)
            print(f"💾 Estado de emergencia guardado")
        raise
    
    # Guardar dataset final
    if main_dataset is not None:
        save_dataset_incremental(main_dataset, dataset_path)
        
        # Limpiar archivos de checkpoint al finalizar exitosamente
        #clean_checkpoint_files(checkpoint_dir)
        
        print(f"✅ Dataset creado exitosamente con {len(main_dataset)} elementos")
        return main_dataset
    else:
        print("❌ No se pudo crear el dataset")
        return Dataset.from_dict({})


def get_song_stream(combined_metadata: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Obtiene el stream de audio principal del chart.
    """
    song_stream_path = ""
    temporary_file = False
    if 'SongStream' in combined_metadata:
        song_stream_path = combined_metadata["SongStream"]
    else:
        available_streams = get_available_streams(combined_metadata)
        if not available_streams:
            raise ValueError("No se encontraron archivos de audio válidos")
        combined_audio = combine_audio_streams(available_streams)
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        combined_audio.export(temp_wav, format="wav")
        temp_wav.close()
        del combined_audio
        song_stream_path = temp_wav.name
        temporary_file = True
    return song_stream_path, temporary_file


def all_streams_available(instruments, streams):
    """
    Verifica si todos los instrumentos necesarios están disponibles.
    """

    instruments_map = {
        'Single': 'guitar.ogg',
        'DoubleRhythm': 'guitar.ogg',
        'DoubleGuitar': 'guitar.ogg',
        'DoubleBass': 'bass.ogg',
        'Drums': 'drums.ogg',
    }

    return all(instruments_map.get(instrument) in streams for instrument in instruments)

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
        instruments = parser.get_available_instruments()
        print("Instrumentos:", instruments)
        difficulties = parser.get_available_difficulties()
        print("Dificultades:", difficulties)
        original_metadata = parser.get_song_metadata()
        stream_metadata = map_streams_to_metadata(streams)
        combined_metadata = {**original_metadata, **stream_metadata}
        song_stream_path, is_temp_file = get_song_stream(combined_metadata)
        separator = None
        if not all_streams_available(instruments, streams):
            separator = DemucsAudioSeparator(song_stream_path, model_name="htdemucs")
        print("Generando embeddings de la canción...")
        song_embeddings = mert(song_stream_path)
    
        bpm, ts, duration, resolution, offset = calculate_bpm_from_folder(parser, streams, song_stream_path)
        song_info = parser.get_song_metadata()
        print("Metadata:", song_info)
        print(f"BPM calculado para '{folder_path}': {bpm}, Time Signature: {ts}, Duración: {duration}, Resolución: {resolution}, Offset: {offset}")
        
        tick_data = song_duration_to_ticks_complete(bpm, duration, resolution, offset)
        tick_ranges = tick_data['ticks']
        all_song_data = []
        for inst, diff in list(product(instruments, difficulties)):
            notes = parser.get_instrument_track_numeric(inst, diff)
            print(f"Notas para {inst} - {diff} (total: {len(notes)}):")
            song_tokens = []
            print("Calculando logmel spectrogram...")
            logmel_spectrogram = calculate_logmel_spectrogram(inst, tick_data["beat_times"], separator, combined_metadata)
            for i in range(len(tick_ranges)-1):
                start_tick = tick_ranges[i]
                end_tick = tick_ranges[i+1]
                notes_in_range = [n for n in notes if start_tick <= n['position'] < end_tick]
                ind_song_token = tokenize_notes(notes_in_range, inst, diff, tokenizer, bpm, resolution, offset, (start_tick, end_tick))
                song_tokens.append([ind_song_token])

            print("Logmel ", logmel_spectrogram.shape)
            print("Tokens ", len(song_tokens))
            data = {
                'logmel_spectrogram': torch.Tensor(logmel_spectrogram).half(),
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
                'song_embeddings': torch.Tensor(song_embeddings).half(),
            }
            all_song_data.append(data)
            
        return all_song_data
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error al calcular BPM: {e}")

    finally:
        if is_temp_file and os.path.exists(song_stream_path):
            os.remove(song_stream_path)
            print(f"Archivo temporal eliminado: {song_stream_path}")



if __name__ == "__main__":
    # Crear dataset de forma incremental
    dataset = create_hf_dataset_incremental(
        folder_path="G:\\Songs",
        process_file_func=process_song,
        batch_size=5,  # Procesar 5 archivos por lote
        save_interval=1,  # Guardar cada 1 archivos
        checkpoint_dir="./checkpoints",
        dataset_path="./clone_hero_dataset_3",
        resume_from_checkpoint=True
    )
    
    print("🎉 ¡Dataset guardado exitosamente!")
    print(f"📊 Total de muestras: {len(dataset)}")
