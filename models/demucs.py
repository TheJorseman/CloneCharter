import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, List
import warnings
import os

# Importar demucs.separate directamente
try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    warnings.warn("Demucs no está instalado. Instálalo con: pip install demucs")

# Importar la clase AudioProcessor anterior
from data.audio_loaders import AudioProcessor

class DemucsAudioSeparator:
    """
    Clase para separar audio usando demucs.separate directamente.
    Genera 3 stems: Guitar + vocals + other, bass y drums en 32 bits.
    """
    
    def __init__(self, 
                 audio_input: Union[str, Path, np.ndarray, torch.Tensor, bytes],
                 original_sample_rate: Optional[int] = None,
                 model_name: str = "htdemucs",
                 device: Optional[str] = None):
        """
        Inicializa el separador de audio con demucs.separate.
        
        Args:
            audio_input: Audio de entrada (path, array, tensor, etc.)
            original_sample_rate: Sample rate original (para arrays/tensors)
            model_name: Modelo de Demucs a usar
            device: Dispositivo ('cpu', 'cuda', None para auto)
        """
        if not DEMUCS_AVAILABLE:
            raise RuntimeError("Demucs no está disponible. Instálalo con: pip install demucs")
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.separated_stems = {}
        self.temp_dir = None
        
        # Procesar audio usando AudioProcessor
        self.audio_processor = AudioProcessor(audio_input, original_sample_rate)
        
        # Separar audio
        self._separate_audio()
        
        # Procesar stems separados
        self._process_separated_stems()
    
    def _separate_audio(self):
        """Separa el audio usando demucs.separate."""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Guardar audio temporal para demucs.separate
            temp_audio_path = Path(self.temp_dir) / "input_audio.wav"
            self.audio_processor.save_resampled_audio(
                temp_audio_path, 
                format='wav', 
                bit_depth=32,
            )
            
            print(f"Separando audio con modelo: {self.model_name}")
            
            # Configurar argumentos para demucs.separate
            args = [
                str(temp_audio_path),
                '-n', self.model_name,
                '-o', str(self.temp_dir),
                '--device', self.device,
                '--float32',  # Usar float32 para 32 bits
                '--mp3-preset', '2'  # Calidad alta si se usa MP3
            ]
            
            # Ejecutar separación usando demucs.separate directamente
            import sys
            original_argv = sys.argv
            try:
                sys.argv = ['demucs.separate'] + args
                demucs.separate.main()
            finally:
                sys.argv = original_argv
            
            print("Separación completada exitosamente")
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Error durante la separación: {e}")
    
    def _process_separated_stems(self):
        """Procesa los stems separados por demucs.separate."""
        # Buscar directorio de salida de demucs.separate
        separated_dir = Path(self.temp_dir) / self.model_name / "input_audio"
        
        if not separated_dir.exists():
            raise RuntimeError(f"No se encontraron stems separados en: {separated_dir}")
        
        # Cargar stems individuales
        stem_files = {
            'vocals': separated_dir / 'vocals.wav',
            'drums': separated_dir / 'drums.wav', 
            'bass': separated_dir / 'bass.wav',
            'other': separated_dir / 'other.wav'
        }
        
        # Verificar que existan los stems necesarios
        loaded_stems = {}
        for stem_name, stem_path in stem_files.items():
            if stem_path.exists():
                try:
                    processor = AudioProcessor(str(stem_path))
                    loaded_stems[stem_name] = processor.get_audio()
                except Exception as e:
                    warnings.warn(f"Error cargando stem {stem_name}: {e}")
                    continue
            else:
                warnings.warn(f"Stem no encontrado: {stem_name}")
        
        # Crear los 3 stems requeridos
        self._create_required_stems(loaded_stems)
    
    def _create_required_stems(self, stems: Dict[str, np.ndarray]):
        """
        Crea los 3 stems requeridos: Guitar + vocals + other, bass y drums.
        
        Args:
            stems: Diccionario con los stems separados por demucs
        """
        # Obtener longitud de referencia
        if not stems:
            raise RuntimeError("No se cargaron stems válidos")
        
        reference_length = len(list(stems.values())[0])
        silence = np.zeros(reference_length, dtype=np.float32)
        
        # 1. Guitar + vocals + other (todo excepto bass y drums)
        guitar_other = silence.copy()
        
        # Sumar vocals y other (guitar está incluido en other)
        #if 'vocals' in stems:
        #    pass
            #guitar_other += stems['vocals']
        if 'other' in stems:
            guitar_other += stems['other']
        
        # 2. Bass
        bass = stems.get('bass', silence.copy())
        
        # 3. Drums
        drums = stems.get('drums', silence.copy())
        
        # Normalizar cada stem para 32 bits
        self.separated_stems = {
            'guitar_other': self._normalize_stem_32bit(guitar_other),
            'bass': self._normalize_stem_32bit(bass),
            'drums': self._normalize_stem_32bit(drums)
        }
        
    
    def _normalize_stem_32bit(self, audio: np.ndarray) -> np.ndarray:
        """
        Normaliza un stem para salida en 32 bits float.
        
        Args:
            audio: Audio a normalizar
            
        Returns:
            Audio normalizado para 32 bits
        """
        if len(audio) == 0:
            return audio
        
        # Convertir a float32
        audio = audio.astype(np.float32)
        
        # Remover DC offset
        audio = audio - np.mean(audio)
        
        # Normalizar al pico máximo con margen mínimo para 32-bit float
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Para 32-bit float, usar normalización agresiva (99% del rango)
            audio = audio * (0.99 / max_val)
        
        return audio
    
    def get_stem(self, stem_name: str, as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Obtiene un stem específico.
        
        Args:
            stem_name: 'guitar_other', 'bass', o 'drums'
            as_tensor: Si retornar como tensor PyTorch
            
        Returns:
            Audio del stem
        """
        valid_stems = ['guitar_other', 'bass', 'drums']
        if stem_name not in valid_stems:
            raise ValueError(f"Stem '{stem_name}' no válido. Válidos: {valid_stems}")
        
        if stem_name not in self.separated_stems:
            raise ValueError(f"Stem '{stem_name}' no disponible")
        
        audio = self.separated_stems[stem_name].copy()
        
        if as_tensor:
            return torch.from_numpy(audio)
        return audio
    
    def get_all_stems(self, as_tensor: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Obtiene los 3 stems.
        
        Args:
            as_tensor: Si retornar como tensores PyTorch
            
        Returns:
            Diccionario con los 3 stems
        """
        result = {}
        for stem_name in ['guitar_other', 'bass', 'drums']:
            if stem_name in self.separated_stems:
                result[stem_name] = self.get_stem(stem_name, as_tensor)
        return result
    
    def save_stem(self, 
                  stem_name: str,
                  output_path: Union[str, Path],
                  format: str = 'wav'):
        """
        Guarda un stem específico en 32 bits.
        
        Args:
            stem_name: Nombre del stem
            output_path: Ruta de salida
            format: Formato de audio
        """
        audio = self.get_stem(stem_name)
        
        # Crear AudioProcessor temporal para guardar en 32 bits
        temp_processor = AudioProcessor(audio, self.audio_processor.target_sample_rate)
        temp_processor.save_resampled_audio(
            output_path, 
            format=format, 
            bit_depth=32, 
        )
    
    def save_all_stems(self, 
                       output_dir: Union[str, Path],
                       format: str = 'wav',
                       prefix: str = ""):
        """
        Guarda los 3 stems en 32 bits.
        
        Args:
            output_dir: Directorio de salida
            format: Formato de audio
            prefix: Prefijo para los nombres de archivo
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem_names = ['guitar_other', 'bass', 'drums']
        
        for stem_name in stem_names:
            if stem_name in self.separated_stems:
                filename = f"{prefix}{stem_name}.{format}" if prefix else f"{stem_name}.{format}"
                output_path = output_dir / filename
                self.save_stem(stem_name, output_path, format)
                print(f"Guardado (32-bit): {output_path}")
    
    def calculate_stem_logmel(self, 
                             stem_name: str,
                             n_mels: int = 512,
                             n_fft: int = 4096,
                             hop_length: int = 1024,
                             as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Calcula el log-mel de un stem específico.
        """
        audio = self.get_stem(stem_name)
        
        # Crear AudioProcessor temporal
        temp_processor = AudioProcessor(audio, self.audio_processor.target_sample_rate)
        
        return temp_processor.calculate_logmel(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            as_tensor=as_tensor
        )
    
    def get_separation_info(self) -> Dict:
        """
        Obtiene información sobre la separación.
        """
        info = self.audio_processor.get_audio_info()
        info.update({
            'model_used': self.model_name,
            'device_used': self.device,
            'available_stems': list(self.separated_stems.keys()),
            'stem_durations': {},
            'bit_depth': 32
        })
        
        for stem_name, audio in self.separated_stems.items():
            duration = len(audio) / self.audio_processor.target_sample_rate
            info['stem_durations'][stem_name] = duration
        
        return info
    
    def _cleanup(self):
        """Limpia archivos temporales."""
        if self.temp_dir and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                warnings.warn(f"No se pudo limpiar directorio temporal: {e}")
    
    def __del__(self):
        """Destructor para limpiar archivos temporales."""
        self._cleanup()

# Función de utilidad para separación rápida
def separate_audio_quick(audio_path: Union[str, Path],
                        output_dir: Union[str, Path],
                        model_name: str = "htdemucs",
                        device: Optional[str] = None) -> DemucsAudioSeparator:
    """
    Función de utilidad para separar audio rápidamente en 3 stems.
    
    Args:
        audio_path: Ruta al archivo de audio
        output_dir: Directorio de salida
        model_name: Modelo de Demucs
        device: Dispositivo a usar
        
    Returns:
        Instancia de DemucsAudioSeparator
    """
    # Separar audio
    separator = DemucsAudioSeparator(audio_path, model_name=model_name, device=device)
    
    # Guardar los 3 stems en 32 bits
    separator.save_all_stems(output_dir, format='wav')
    
    return separator