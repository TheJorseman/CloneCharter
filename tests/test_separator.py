from models.demucs import DemucsAudioSeparator, separate_audio_quick

# Ejemplo de uso
if __name__ == "__main__":
    # Separar audio desde archivo
    separator = DemucsAudioSeparator("Sera_porque_te_amo.mp3", model_name="htdemucs")
    
    # Guardar los 3 stems en 32 bits
    separator.save_all_stems("output_stems/")
    
    # Obtener stems individuales
    guitar_vocals = separator.get_stem("guitar_other")
    bass = separator.get_stem("bass")
    drums = separator.get_stem("drums")
    
    print(f"Guitar+ Other shape: {guitar_vocals.shape}")
    print(f"Bass shape: {bass.shape}")
    print(f"Drums shape: {drums.shape}")
    
    # Calcular log-mel de un stem
    bass_logmel = separator.calculate_stem_logmel("bass", n_mels=80)
    print(f"Bass logmel shape: {bass_logmel.shape}")
    
    # Información de la separación
    info = separator.get_separation_info()
    print(f"Información: {info}")
    
    # Usar función rápida
    #separator2 = separate_audio_quick("another_song.wav", "output_stems2/")
