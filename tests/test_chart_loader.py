from data.chart_loader import CloneHeroChartParser
from utils.audio_utils import procesar_data_musical
from data.audio_loaders import AudioProcessor

# Cargar archivo chart
with open('notes.chart', 'r', encoding='utf-8') as f:
    chart_content = f.read()

# Crear parser
parser = CloneHeroChartParser(chart_content)

# Obtener todas las secciones
print("Secciones encontradas:", parser.get_sections())

# Obtener metadatos de la canción
song_info = parser.get_song_metadata()
print(song_info)

# Obtener track de guitarra expert
guitar_notes = parser.get_instrument_track_numeric('Single', 'Expert')
#print(guitar_notes)

# Instrumentos disponibles
instruments = parser.get_available_instruments()
print("Instrumentos:", instruments)

# Dificultades disponibles  
difficulties = parser.get_available_difficulties()
print("Dificultades:", difficulties)
# Obtener datos estructurados
summary = parser.get_expert_to_easy_summary()
#print(summary)

#print("Datos de la canción:", parser.get_sync_track())

file = "./Grupo Marca Registrada - El Rescate/vocals.ogg"
processor = AudioProcessor(file)

duration = processor.get_audio_info().get('original_duration', 1)
print(duration)
resolucion = song_info.get("Resolution", 192)  # Resolución por defecto
offset = song_info.get("Offset", 0.0)  # Offset por defecto
#data = parser.get_sync_track()


guitar_notes = parser.get_instrument_track_numeric('Single', 'Expert')




#resultado = procesar_data_musical(duracion=duration, resolucion=resolucion, offset=offset, data=data)

#print(resultado)


