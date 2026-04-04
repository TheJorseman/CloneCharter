import numpy as np
import torch
import librosa
import soundfile as sf
import audiofile
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings
import io
import wave

class AudioProcessor:
    """
    Clase para procesar archivos de audio con capacidades de resample, extracción de logmel y guardado.
    Compatible con múltiples formatos de entrada y salida, multiplataforma.
    """
    
    # Sample rate de Demucs (44.1 kHz)
    DEMUCS_SAMPLE_RATE = 44100
    
    def __init__(self, 
                 audio_input: Union[str, Path, np.ndarray, torch.Tensor, bytes],
                 original_sample_rate: Optional[int] = None):
        """
        Inicializa el procesador de audio.
        
        Args:
            audio_input: Puede ser un path al archivo, buffer de bytes, array numpy o tensor pytorch
            original_sample_rate: Sample rate original (requerido para arrays/tensors)
        """
        self.original_sample_rate = original_sample_rate
        self.target_sample_rate = self.DEMUCS_SAMPLE_RATE
        self.audio_data = None
        self.resampled_audio = None
        
        self._load_audio(audio_input)
        self._resample_to_demucs()
    
    def _load_audio(self, audio_input: Union[str, Path, np.ndarray, torch.Tensor, bytes]):
        """Determina el tipo de entrada y llama a la función de carga correspondiente."""
        
        if isinstance(audio_input, (str, Path)):
            self._load_from_file(audio_input)
        elif isinstance(audio_input, bytes):
            self._load_from_buffer(audio_input)
        elif isinstance(audio_input, np.ndarray):
            self._load_from_numpy(audio_input)
        elif isinstance(audio_input, torch.Tensor):
            self._load_from_tensor(audio_input)
        else:
            raise ValueError(f"Tipo de entrada no soportado: {type(audio_input)}")
        
        self._normalize_audio()
    
    def _load_from_file(self, file_path: Union[str, Path]):
        """
        Carga audio desde un archivo.
        
        Args:
            file_path: Ruta al archivo de audio
        """
        try:
            # Usar audiofile como primera opción (multiplataforma)
            self.audio_data, self.original_sample_rate = audiofile.read(str(file_path))
            # Convertir a mono si es estéreo
            if len(self.audio_data.shape) > 1 and self.audio_data.shape[0] > 1:
                self.audio_data = np.mean(self.audio_data, axis=0)
        except Exception as e:
            # Fallback a librosa si audiofile falla
            warnings.warn(f"audiofile falló, usando librosa: {e}")
            try:
                self.audio_data, self.original_sample_rate = librosa.load(
                    str(file_path), sr=None, mono=True
                )
            except Exception as e2:
                raise ValueError(f"No se pudo cargar el archivo de audio: {e2}")
    
    def _load_from_buffer(self, buffer: bytes):
        """
        Carga audio desde un buffer de bytes.
        
        Args:
            buffer: Buffer de bytes conteniendo datos de audio
        """
        try:
            # Usar soundfile para leer desde buffer
            self.audio_data, self.original_sample_rate = sf.read(io.BytesIO(buffer))
            # Convertir a mono si es estéreo
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
        except Exception as e:
            # Intentar con librosa como fallback
            try:
                # Guardar temporalmente el buffer y cargar con librosa
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(buffer)
                    tmp_file.flush()
                    self.audio_data, self.original_sample_rate = librosa.load(
                        tmp_file.name, sr=None, mono=True
                    )
                # Limpiar archivo temporal
                import os
                os.unlink(tmp_file.name)
            except Exception as e2:
                raise ValueError(f"No se pudo cargar audio desde buffer: {e} | {e2}")
    
    def _load_from_numpy(self, array: np.ndarray):
        """
        Carga audio desde un array de numpy.
        
        Args:
            array: Array numpy conteniendo datos de audio
        """
        if self.original_sample_rate is None:
            raise ValueError("Se requiere original_sample_rate para arrays numpy")
        
        self.audio_data = array.copy()
        
        # Asegurar que sea 1D (convertir a mono si es necesario)
        if len(self.audio_data.shape) > 1:
            if self.audio_data.shape[0] <= 2:  # Formato (canales, muestras)
                self.audio_data = np.mean(self.audio_data, axis=0)
            else:  # Formato (muestras, canales)
                self.audio_data = np.mean(self.audio_data, axis=1)
    
    def _load_from_tensor(self, tensor: torch.Tensor):
        """
        Carga audio desde un tensor de pytorch.
        
        Args:
            tensor: Tensor pytorch conteniendo datos de audio
        """
        if self.original_sample_rate is None:
            raise ValueError("Se requiere original_sample_rate para tensors pytorch")
        
        # Convertir tensor a numpy
        self.audio_data = tensor.detach().cpu().numpy()
        
        # Asegurar que sea 1D (convertir a mono si es necesario)
        if len(self.audio_data.shape) > 1:
            if self.audio_data.shape[0] <= 2:  # Formato (canales, muestras)
                self.audio_data = np.mean(self.audio_data, axis=0)
            else:  # Formato (muestras, canales)
                self.audio_data = np.mean(self.audio_data, axis=1)
    
    def _normalize_audio(self):
        """Normaliza el audio a float32 y escala entre -1 y 1."""
        self.audio_data = self.audio_data.astype(np.float32)
        
        # Normalizar solo si hay señal
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            self.audio_data = self.audio_data / max_val
    
    def _resample_to_demucs(self):
        """Resamplea el audio al sample rate de Demucs usando resampy para mejor calidad."""
        if self.original_sample_rate == self.target_sample_rate:
            self.resampled_audio = self.audio_data.copy()
        else:
            try:
                # Usar resampy para mejor calidad en resampling
                import resampy
                self.resampled_audio = resampy.resample(
                    self.audio_data, 
                    self.original_sample_rate, 
                    self.target_sample_rate,
                    filter='kaiser_best'
                )
            except ImportError:
                # Fallback a librosa si resampy no está disponible
                warnings.warn("resampy no disponible, usando librosa.resample")
                self.resampled_audio = librosa.resample(
                    self.audio_data,
                    orig_sr=self.original_sample_rate,
                    target_sr=self.target_sample_rate
                )
    
    def get_audio(self, as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Obtiene el audio resampleado.
        
        Args:
            as_tensor: Si True, retorna como tensor pytorch
            
        Returns:
            Audio resampleado como numpy array o tensor pytorch
        """
        if as_tensor:
            return torch.from_numpy(self.resampled_audio)
        return self.resampled_audio.copy()
    
    def calculate_logmel(self, 
                        n_mels: int = 128,
                        n_fft: int = 2048,
                        hop_length: int = 512,
                        fmin: float = 0.0,
                        fmax: Optional[float] = None,
                        as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Calcula el espectrograma log-mel del audio.
        
        Args:
            n_mels: Número de bandas mel
            n_fft: Tamaño de la FFT
            hop_length: Número de muestras entre frames
            fmin: Frecuencia mínima
            fmax: Frecuencia máxima (None para sr/2)
            as_tensor: Si True, retorna como tensor pytorch
            
        Returns:
            Espectrograma log-mel
        """
        if fmax is None:
            fmax = self.target_sample_rate // 2
        
        # Calcular mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=self.resampled_audio,
            sr=self.target_sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        
        # Convertir a escala logarítmica
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        if as_tensor:
            return torch.from_numpy(log_mel)
        return log_mel
    
    def _normalize_for_export(self, audio_data: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Normaliza el audio específicamente para exportación con volumen adecuado.
        
        Args:
            audio_data: Datos de audio originales
            target_db: Nivel objetivo en dB (por defecto -3dB para evitar clipping)
            
        Returns:
            Audio normalizado con volumen apropiado
        """
        # Crear copia para no modificar original
        audio = audio_data.copy().astype(np.float64)
        
        # Remover DC offset
        audio = audio - np.mean(audio)
        
        # Calcular RMS (Root Mean Square) para normalización por volumen percibido
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Convertir target_db a factor lineal
            target_linear = 10**(target_db/20.0)
            
            # Normalizar basado en RMS para mejor volumen percibido
            audio = audio * (target_linear / rms)
            
            # Verificar que no hay clipping y ajustar si es necesario
            max_val = np.max(np.abs(audio))
            if max_val > 0.95:  # Dejar margen para evitar clipping
                audio = audio * (0.95 / max_val)
        
        # Clamp final para asegurar rango válido
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio.astype(np.float32)

    def _save_wav_improved(self, 
                        audio_data: np.ndarray,
                        sample_rate: int,
                        output_path: Path,
                        bit_depth: int = 16):
        """
        Guarda audio en formato WAV con volumen corregido.
        
        Args:
            audio_data: Datos de audio normalizados
            sample_rate: Sample rate
            output_path: Ruta de salida
            bit_depth: Profundidad de bits
        """
        try:
            # Usar soundfile para WAV con mejor manejo de volumen
            if bit_depth == 16:
                # Para 16-bit, usar el rango completo sin pérdida de volumen
                max_int16 = 32767
                audio_int = np.round(audio_data * max_int16).astype(np.int16)
                sf.write(str(output_path), audio_int, sample_rate, subtype='PCM_16')
                
            elif bit_depth == 24:
                # Para 24-bit, usar el rango completo
                max_int24 = 8388607  # 2^23 - 1
                audio_int = np.round(audio_data * max_int24).astype(np.int32)
                sf.write(str(output_path), audio_int, sample_rate, subtype='PCM_24')
                
            elif bit_depth == 32:
                # Para 32-bit float, mantener como float32
                sf.write(str(output_path), audio_data.astype(np.float32), sample_rate, subtype='FLOAT')
                
            else:
                # Default a 32-bit con volumen completo
                sf.write(str(output_path), audio_data.astype(np.float32), sample_rate, subtype='FLOAT')
                
        except Exception as e:
            # Fallback usando wave nativo con volumen corregido
            self._save_wav_native_improved(audio_data, sample_rate, output_path, bit_depth)

    def _save_wav_native_improved(self, 
                                audio_data: np.ndarray,
                                sample_rate: int,
                                output_path: Path,
                                bit_depth: int = 16):
        """
        Guarda WAV usando la librería wave nativa con volumen corregido.
        
        Args:
            audio_data: Datos de audio
            sample_rate: Sample rate
            output_path: Ruta de salida
            bit_depth: Profundidad de bits
        """
        if bit_depth == 16:
            max_val = 32767
            audio_int = np.round(audio_data * max_val).astype(np.int16)
            sample_width = 2
            
        elif bit_depth == 24:
            max_val = 8388607  # 2^23 - 1
            audio_int = np.round(audio_data * max_val).astype(np.int32)
            
            # Convertir a bytes de 24-bit manualmente
            audio_bytes = bytearray()
            for sample in audio_int:
                # Asegurar que está en el rango correcto
                sample = np.clip(sample, -8388608, 8388607)
                # Convertir a 24-bit little-endian
                if sample < 0:
                    sample = sample + 16777216  # 2^24
                
                audio_bytes.extend([
                    sample & 0xFF,
                    (sample >> 8) & 0xFF,
                    (sample >> 16) & 0xFF
                ])
            
            audio_int = bytes(audio_bytes)
            sample_width = 3
            
        else:  # Default 16-bit
            max_val = 32767
            audio_int = np.round(audio_data * max_val).astype(np.int16)
            sample_width = 2
        
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            
            if isinstance(audio_int, bytes):
                wf.writeframes(audio_int)
            else:
                wf.writeframes(audio_int.tobytes())

    # Función adicional para ajustar volumen manualmente
    def adjust_volume(self, audio_data: np.ndarray, db_change: float) -> np.ndarray:
        """
        Ajusta el volumen del audio en decibeles.
        
        Args:
            audio_data: Datos de audio
            db_change: Cambio en dB (positivo para aumentar, negativo para disminuir)
            
        Returns:
            Audio con volumen ajustado
        """
        # Convertir dB a factor lineal
        linear_factor = 10**(db_change/20.0)
        
        # Aplicar factor
        adjusted_audio = audio_data * linear_factor
        
        # Prevenir clipping
        max_val = np.max(np.abs(adjusted_audio))
        if max_val > 1.0:
            adjusted_audio = adjusted_audio / max_val
            warnings.warn(f"Audio recortado para evitar clipping. Factor aplicado: {1.0/max_val:.3f}")
        
        return adjusted_audio

    # Métodos actualizados para la clase
    def save_original_audio(self, 
                        output_path: Union[str, Path], 
                        format: str = 'wav',
                        bit_depth: int = 16,
                        volume_boost_db: float = 0.0):
        """
        Guarda el audio original con volumen corregido.
        
        Args:
            output_path: Ruta de salida
            format: Formato de audio ('wav', 'flac', 'mp3', 'ogg')
            bit_depth: Profundidad de bits (16, 24, 32)
            volume_boost_db: Boost adicional en dB (ej: 6.0 para +6dB)
        """
        audio_to_save = self.audio_data.copy()
        
        # Aplicar boost de volumen si se especifica
        if volume_boost_db != 0.0:
            audio_to_save = self.adjust_volume(audio_to_save, volume_boost_db)
        
        # Normalizar para exportación con volumen apropiado
        audio_normalized = self._normalize_for_export(audio_to_save, target_db=-3.0)
        
        self._save_audio_data(
            audio_normalized, 
            self.original_sample_rate, 
            output_path, 
            format, 
            bit_depth
        )

    def save_resampled_audio(self, 
                            output_path: Union[str, Path], 
                            format: str = 'wav',
                            bit_depth: int = 16,
                            volume_boost_db: float = 0.0):
        """
        Guarda el audio resampleado con volumen corregido.
        
        Args:
            output_path: Ruta de salida
            format: Formato de audio ('wav', 'flac', 'mp3', 'ogg')
            bit_depth: Profundidad de bits (16, 24, 32)
            volume_boost_db: Boost adicional en dB (ej: 6.0 para +6dB)
        """
        audio_to_save = self.resampled_audio.copy()
        
        # Aplicar boost de volumen si se especifica
        if volume_boost_db != 0.0:
            audio_to_save = self.adjust_volume(audio_to_save, volume_boost_db)
        
        # Normalizar para exportación con volumen apropiado
        audio_normalized = self._normalize_for_export(audio_to_save, target_db=-3.0)
        
        self._save_audio_data(
            audio_normalized, 
            self.target_sample_rate, 
            output_path, 
            format, 
            bit_depth
        )

    
    def save_audio_segment(self, 
                          start_time: float, 
                          end_time: float,
                          output_path: Union[str, Path],
                          use_resampled: bool = True,
                          format: str = 'wav',
                          bit_depth: int = 16):
        """
        Guarda un segmento específico del audio.
        
        Args:
            start_time: Tiempo de inicio en segundos
            end_time: Tiempo de fin en segundos
            output_path: Ruta de salida
            use_resampled: Si usar audio resampleado o original
            format: Formato de audio
            bit_depth: Profundidad de bits
        """
        if use_resampled:
            audio_data = self.resampled_audio
            sample_rate = self.target_sample_rate
        else:
            audio_data = self.audio_data
            sample_rate = self.original_sample_rate
        
        # Calcular índices de muestras
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Validar límites
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            raise ValueError("Tiempo de inicio debe ser menor que tiempo de fin")
        
        # Extraer segmento
        segment = audio_data[start_sample:end_sample]
        
        # Guardar segmento
        self._save_audio_data(segment, sample_rate, output_path, format, bit_depth)
    
    def _save_audio_data(self, 
                        audio_data: np.ndarray,
                        sample_rate: int,
                        output_path: Union[str, Path],
                        format: str = 'wav',
                        bit_depth: int = 16):
        """
        Función interna mejorada para guardar datos de audio.
        
        Args:
            audio_data: Datos de audio a guardar
            sample_rate: Sample rate del audio
            output_path: Ruta de salida
            format: Formato de audio
            bit_depth: Profundidad de bits
        """
        output_path = Path(output_path)
        format = format.lower()
        
        # Asegurar que la extensión coincida con el formato
        if not output_path.suffix:
            output_path = output_path.with_suffix(f'.{format}')
        
        # Normalizar audio para evitar clipping y ruido
        audio_normalized = self._normalize_for_export(audio_data)
        
        try:
            if format == 'wav':
                self._save_wav_improved(audio_normalized, sample_rate, output_path, bit_depth)
            elif format == 'ogg':
                self._save_ogg_improved(audio_normalized, sample_rate, output_path)
            else:
                # Para otros formatos, usar soundfile con configuración específica
                self._save_with_soundfile(audio_normalized, sample_rate, output_path, format, bit_depth)
        except Exception as e:
            warnings.warn(f"Error al guardar con método principal: {e}")
            # Fallback mejorado
            self._save_fallback(audio_normalized, sample_rate, output_path, format, bit_depth)


    def _save_ogg_improved(self, 
                        audio_data: np.ndarray,
                        sample_rate: int,
                        output_path: Path):
        """
        Guarda audio en formato OGG con configuración optimizada.
        
        Args:
            audio_data: Datos de audio normalizados
            sample_rate: Sample rate
            output_path: Ruta de salida
        """
        try:
            # Intentar con pydub primero (mejor para OGG)
            from pydub import AudioSegment
            
            # Convertir a 16-bit para pydub
            audio_int16 = np.round(audio_data * 32767).astype(np.int16)
            
            # Crear AudioSegment
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes
                channels=1
            )
            
            # Exportar con configuración optimizada
            audio_segment.export(
                str(output_path),
                format="ogg",
                codec="libvorbis",
                parameters=["-q:a", "5"]  # Calidad media-alta
            )
            
        except ImportError:
            # Si pydub no está disponible, usar soundfile
            try:
                sf.write(str(output_path), audio_data, sample_rate, format='OGG', subtype='VORBIS')
            except Exception as e:
                raise ValueError(f"No se pudo guardar archivo OGG. Instala pydub: pip install pydub. Error: {e}")
        except Exception as e:
            # Fallback a soundfile
            try:
                sf.write(str(output_path), audio_data, sample_rate, format='OGG', subtype='VORBIS')
            except Exception as e2:
                raise ValueError(f"Error al guardar OGG: {e} | {e2}")

    def _save_with_soundfile(self, 
                            audio_data: np.ndarray,
                            sample_rate: int,
                            output_path: Path,
                            format: str,
                            bit_depth: int):
        """
        Guarda usando soundfile con configuración específica por formato.
        
        Args:
            audio_data: Datos de audio
            sample_rate: Sample rate
            output_path: Ruta de salida
            format: Formato de audio
            bit_depth: Profundidad de bits
        """
        format_map = {
            'flac': {'format': 'FLAC', 'subtype': 'PCM_16' if bit_depth == 16 else 'PCM_24'},
            'mp3': {'format': 'MP3', 'subtype': None},
            'aiff': {'format': 'AIFF', 'subtype': 'PCM_16' if bit_depth == 16 else 'PCM_24'},
            'au': {'format': 'AU', 'subtype': 'PCM_16'}
        }
        
        if format in format_map:
            config = format_map[format]
            if config['subtype']:
                sf.write(str(output_path), audio_data, sample_rate, 
                        format=config['format'], subtype=config['subtype'])
            else:
                sf.write(str(output_path), audio_data, sample_rate, format=config['format'])
        else:
            # Formato genérico
            sf.write(str(output_path), audio_data, sample_rate)

    def _save_fallback(self, 
                    audio_data: np.ndarray,
                    sample_rate: int,
                    output_path: Path,
                    format: str,
                    bit_depth: int):
        """
        Método de respaldo usando diferentes librerías.
        
        Args:
            audio_data: Datos de audio
            sample_rate: Sample rate
            output_path: Ruta de salida
            format: Formato de audio
            bit_depth: Profundidad de bits
        """
        try:
            # Intentar con librosa
            if format == 'wav':
                # Convertir a entero apropiado
                if bit_depth == 16:
                    audio_int = np.round(audio_data * 32767).astype(np.int16)
                else:
                    audio_int = audio_data
                
                librosa.output.write_wav(str(output_path), audio_int, sample_rate)
            else:
                # Para otros formatos, convertir a WAV como último recurso
                wav_path = output_path.with_suffix('.wav')
                audio_int = np.round(audio_data * 32767).astype(np.int16)
                librosa.output.write_wav(str(wav_path), audio_int, sample_rate)
                warnings.warn(f"Guardado como WAV en lugar de {format}: {wav_path}")
                
        except Exception as e:
            # Último recurso: usar wave nativo para WAV
            if format == 'wav' or True:  # Forzar WAV como último recurso
                wav_path = output_path.with_suffix('.wav')
                self._save_wav_native(audio_data, sample_rate, wav_path, bit_depth)
                if format != 'wav':
                    warnings.warn(f"Guardado como WAV en lugar de {format}: {wav_path}")
            else:
                raise ValueError(f"No se pudo guardar el archivo en ningún formato: {e}")

    def _save_wav_native(self, 
                        audio_data: np.ndarray,
                        sample_rate: int,
                        output_path: Path,
                        bit_depth: int = 16):
        """
        Guarda WAV usando la librería wave nativa de Python.
        
        Args:
            audio_data: Datos de audio
            sample_rate: Sample rate
            output_path: Ruta de salida
            bit_depth: Profundidad de bits
        """
        # Convertir a entero según bit depth
        if bit_depth == 16:
            audio_int = np.round(audio_data * 32767).astype(np.int16)
            sample_width = 2
        elif bit_depth == 24:
            audio_int = np.round(audio_data * 8388607).astype(np.int32)
            # Para 24-bit, necesitamos manejar los bytes manualmente
            audio_bytes = []
            for sample in audio_int:
                # Convertir a 24-bit little-endian
                sample_24 = sample & 0xFFFFFF
                audio_bytes.extend([
                    sample_24 & 0xFF,
                    (sample_24 >> 8) & 0xFF,
                    (sample_24 >> 16) & 0xFF
                ])
            audio_int = bytes(audio_bytes)
            sample_width = 3
        else:
            audio_int = np.round(audio_data * 32767).astype(np.int16)
            sample_width = 2
        
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            
            if isinstance(audio_int, bytes):
                wf.writeframes(audio_int)
            else:
                wf.writeframes(audio_int.tobytes())
    
    def get_audio_info(self) -> dict:
        """
        Obtiene información del audio procesado.
        
        Returns:
            Diccionario con información del audio
        """
        return {
            'original_sample_rate': self.original_sample_rate,
            'target_sample_rate': self.target_sample_rate,
            'original_duration': len(self.audio_data) / self.original_sample_rate,
            'resampled_duration': len(self.resampled_audio) / self.target_sample_rate,
            'original_samples': len(self.audio_data),
            'resampled_samples': len(self.resampled_audio),
            'channels': 1  # Siempre mono después del procesamiento
        }

# Funciones de utilidad independientes
def save_numpy_as_audio(audio_array: np.ndarray,
                       sample_rate: int,
                       output_path: Union[str, Path],
                       format: str = 'wav',
                       bit_depth: int = 16):
    """
    Guarda un array numpy directamente como archivo de audio.
    
    Args:
        audio_array: Array numpy con datos de audio
        sample_rate: Sample rate del audio
        output_path: Ruta de salida
        format: Formato de audio
        bit_depth: Profundidad de bits
    """
    processor = AudioProcessor.__new__(AudioProcessor)
    processor._save_audio_data(audio_array, sample_rate, output_path, format, bit_depth)

def save_tensor_as_audio(audio_tensor: torch.Tensor,
                        sample_rate: int,
                        output_path: Union[str, Path],
                        format: str = 'wav',
                        bit_depth: int = 16):
    """
    Guarda un tensor pytorch directamente como archivo de audio.
    
    Args:
        audio_tensor: Tensor pytorch con datos de audio
        sample_rate: Sample rate del audio
        output_path: Ruta de salida
        format: Formato de audio
        bit_depth: Profundidad de bits
    """
    audio_array = audio_tensor.detach().cpu().numpy()
    save_numpy_as_audio(audio_array, sample_rate, output_path, format, bit_depth)
