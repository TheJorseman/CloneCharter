from data.midi_loader import MIDIProcessor
# Ejemplo de uso

if __name__ == "__main__":
    # Cargar un archivo existente y modificarlo
    processor = MIDIProcessor("notes.mid")
    stats = processor.obtener_estadisticas()
    print("\nEstadísticas del archivo:")
    for clave, valor in stats.items():
        print(f"{clave}: {valor}")

    # Transponer la primera pista 2 semitonos hacia arriba
    processor.transponer_notas(0, 2)

    # Cambiar el instrumento de la segunda pista a violín
    processor.cambiar_instrumento(1, 40)

    # Aumentar la velocidad de todas las notas en 20%
    processor.cambiar_velocidad_global(0, 1.2)

    # Guardar con un nuevo nombre
    processor.guardar_archivo("mi_cancion_modificada.mid")


