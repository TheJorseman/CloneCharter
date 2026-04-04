from data.audio_loaders import AudioProcessor, save_numpy_as_audio, save_tensor_as_audio
import numpy as np
import torch
import os
# Ejemplo de uso
if __name__ == "__main__":
    folder = "./test/audio/"
    os.makedirs(folder, exist_ok=True)
    # Cargar y procesar audio
    processor = AudioProcessor("guitar.ogg")
    
    # Guardar con volumen normal (normalizado a -3dB)
    processor.save_original_audio(os.path.join(folder, "output_original.wav"), format='mp3', bit_depth=32)
    processor.save_resampled_audio(os.path.join(folder, "output_resampled.wav"), format='wav', bit_depth=32)

    # Guardar con boost adicional de volumen (+6dB)
    processor.save_original_audio(os.path.join(folder, "output_original_loud.wav"), format='wav', bit_depth=32, volume_boost_db=6.0)
    processor.save_resampled_audio(os.path.join(folder, "output_resampled_loud.wav"), format='wav', bit_depth=32, volume_boost_db=6.0)

    # Para casos extremos de volumen muy bajo, usar boost mayor
    processor.save_resampled_audio(os.path.join(folder,"output_very_loud.wav"), format='wav', bit_depth=32, volume_boost_db=12.0)
    
    # Guardar segmento específico (de 5 a 10 segundos)
    processor.save_audio_segment(
        start_time=5.0, 
        end_time=10.0, 
        output_path=os.path.join(folder,"segment.wav"),
        use_resampled=True
    )
    
    # Guardar en diferentes formatos
    processor.save_resampled_audio(os.path.join(folder,"output.flac"), format='flac', bit_depth=24)
    processor.save_resampled_audio(os.path.join(folder,"output.ogg"), format='ogg', bit_depth=24)
    
    # Usar funciones independientes
    audio_array = np.random.randn(44100 * 2)  # 2 segundos
    save_numpy_as_audio(audio_array, 44100, os.path.join(folder,"from_numpy.wav"))
    
    audio_tensor = torch.randn(44100 * 3)  # 3 segundos
    save_tensor_as_audio(audio_tensor, 44100, os.path.join(folder,"from_tensor.wav"))
    
    print("Archivos de audio guardados exitosamente")