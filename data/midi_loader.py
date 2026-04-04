import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import os

class MIDIProcessor:
    def __init__(self, archivo_midi=None):
        """
        Inicializa el procesador MIDI
        
        Args:
            archivo_midi (str): Ruta del archivo MIDI a cargar (opcional)
        """
        self.midi_file = None
        self.archivo_original = None
        
        if archivo_midi:
            self.cargar_archivo(archivo_midi)
    
    def cargar_archivo(self, ruta_archivo):
        """
        Carga un archivo MIDI desde disco
        
        Args:
            ruta_archivo (str): Ruta del archivo MIDI
        """
        try:
            self.midi_file = MidiFile(ruta_archivo)
            self.archivo_original = ruta_archivo
            print(f"Archivo MIDI cargado: {ruta_archivo}")
            self._mostrar_info_basica()
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
    
    def _mostrar_info_basica(self):
        """Muestra información básica del archivo MIDI"""
        if self.midi_file:
            print(f"Tipo: {self.midi_file.type}")
            print(f"Número de pistas: {len(self.midi_file.tracks)}")
            print(f"Ticks por beat: {self.midi_file.ticks_per_beat}")
            print(f"Duración: {self.midi_file.length:.2f} segundos")
    
    def crear_archivo_nuevo(self, tipo=1, ticks_per_beat=480):
        """
        Crea un nuevo archivo MIDI vacío
        
        Args:
            tipo (int): Tipo de archivo MIDI (0, 1, o 2)
            ticks_per_beat (int): Resolución temporal
        """
        self.midi_file = MidiFile(type=tipo, ticks_per_beat=ticks_per_beat)
        print(f"Nuevo archivo MIDI creado (tipo {tipo})")
    
    def agregar_pista(self, nombre_pista=""):
        """
        Agrega una nueva pista al archivo MIDI
        
        Args:
            nombre_pista (str): Nombre de la pista
            
        Returns:
            int: Índice de la nueva pista
        """
        if not self.midi_file:
            self.crear_archivo_nuevo()
        
        nueva_pista = MidiTrack()
        
        # Agregar nombre de pista si se proporciona
        if nombre_pista:
            nueva_pista.append(MetaMessage('track_name', name=nombre_pista, time=0))
        
        self.midi_file.tracks.append(nueva_pista)
        indice_pista = len(self.midi_file.tracks) - 1
        print(f"Pista agregada: {nombre_pista} (índice {indice_pista})")
        return indice_pista
    
    def agregar_nota(self, pista, nota, velocidad, tiempo_inicio, duracion, canal=0):
        """
        Agrega una nota a una pista específica
        
        Args:
            pista (int): Índice de la pista
            nota (int): Número MIDI de la nota (0-127)
            velocidad (int): Velocidad de la nota (0-127)
            tiempo_inicio (int): Tiempo de inicio en ticks
            duracion (int): Duración en ticks
            canal (int): Canal MIDI (0-15)
        """
        if not self.midi_file or pista >= len(self.midi_file.tracks):
            print("Error: Pista no válida")
            return
        
        track = self.midi_file.tracks[pista]
        
        # Nota encendida
        note_on = Message('note_on', channel=canal, note=nota, 
                         velocity=velocidad, time=tiempo_inicio)
        track.append(note_on)
        
        # Nota apagada
        note_off = Message('note_off', channel=canal, note=nota, 
                          velocity=velocidad, time=duracion)
        track.append(note_off)
    
    def cambiar_tempo(self, pista, nuevo_tempo, tiempo=0):
        """
        Cambia el tempo en una pista
        
        Args:
            pista (int): Índice de la pista
            nuevo_tempo (int): Nuevo tempo en BPM
            tiempo (int): Tiempo en ticks donde aplicar el cambio
        """
        if not self.midi_file or pista >= len(self.midi_file.tracks):
            print("Error: Pista no válida")
            return
        
        # Convertir BPM a microsegundos por beat
        microsegundos_por_beat = int(60000000 / nuevo_tempo)
        
        tempo_msg = MetaMessage('set_tempo', tempo=microsegundos_por_beat, time=tiempo)
        self.midi_file.tracks[pista].append(tempo_msg)
        print(f"Tempo cambiado a {nuevo_tempo} BPM en pista {pista}")
    
    def cambiar_instrumento(self, pista, programa, canal=0, tiempo=0):
        """
        Cambia el instrumento (programa) en una pista
        
        Args:
            pista (int): Índice de la pista
            programa (int): Número de programa (0-127)
            canal (int): Canal MIDI (0-15)
            tiempo (int): Tiempo en ticks
        """
        if not self.midi_file or pista >= len(self.midi_file.tracks):
            print("Error: Pista no válida")
            return
        
        program_change = Message('program_change', channel=canal, 
                               program=programa, time=tiempo)
        self.midi_file.tracks[pista].append(program_change)
        print(f"Instrumento cambiado a programa {programa} en pista {pista}")
    
    def transponer_notas(self, pista, semitonos):
        """
        Transpone todas las notas de una pista
        
        Args:
            pista (int): Índice de la pista
            semitonos (int): Número de semitonos a transponer (+ o -)
        """
        if not self.midi_file or pista >= len(self.midi_file.tracks):
            print("Error: Pista no válida")
            return
        
        track = self.midi_file.tracks[pista]
        notas_modificadas = 0
        
        for mensaje in track:
            if mensaje.type in ['note_on', 'note_off']:
                nueva_nota = mensaje.note + semitonos
                # Asegurar que la nota esté en el rango válido (0-127)
                if 0 <= nueva_nota <= 127:
                    mensaje.note = nueva_nota
                    notas_modificadas += 1
        
        print(f"Transposición completada: {notas_modificadas} notas modificadas en {semitonos} semitonos")
    
    def cambiar_velocidad_global(self, pista, factor):
        """
        Cambia la velocidad de todas las notas en una pista
        
        Args:
            pista (int): Índice de la pista
            factor (float): Factor de multiplicación para la velocidad
        """
        if not self.midi_file or pista >= len(self.midi_file.tracks):
            print("Error: Pista no válida")
            return
        
        track = self.midi_file.tracks[pista]
        notas_modificadas = 0
        
        for mensaje in track:
            if mensaje.type == 'note_on' and mensaje.velocity > 0:
                nueva_velocidad = int(mensaje.velocity * factor)
                # Asegurar que la velocidad esté en el rango válido (1-127)
                mensaje.velocity = max(1, min(127, nueva_velocidad))
                notas_modificadas += 1
        
        print(f"Velocidad modificada: {notas_modificadas} notas afectadas")
    
    def obtener_estadisticas(self):
        """
        Obtiene estadísticas del archivo MIDI
        
        Returns:
            dict: Diccionario con estadísticas
        """
        if not self.midi_file:
            return {}
        
        stats = {
            'numero_pistas': len(self.midi_file.tracks),
            'duracion_segundos': self.midi_file.length,
            'ticks_per_beat': self.midi_file.ticks_per_beat,
            'tipo': self.midi_file.type,
            'total_mensajes': sum(len(track) for track in self.midi_file.tracks)
        }
        
        return stats
    
    def listar_mensajes_pista(self, pista, limite=10):
        """
        Lista los mensajes de una pista específica
        
        Args:
            pista (int): Índice de la pista
            limite (int): Número máximo de mensajes a mostrar
        """
        if not self.midi_file or pista >= len(self.midi_file.tracks):
            print("Error: Pista no válida")
            return
        
        track = self.midi_file.tracks[pista]
        print(f"\nMensajes de la pista {pista} (mostrando {min(limite, len(track))}):")
        
        for i, mensaje in enumerate(track[:limite]):
            print(f"{i}: {mensaje}")
        
        if len(track) > limite:
            print(f"... y {len(track) - limite} mensajes más")
    
    def guardar_archivo(self, ruta_salida=None):
        """
        Guarda el archivo MIDI
        
        Args:
            ruta_salida (str): Ruta donde guardar el archivo. Si es None, sobrescribe el original
        """
        if not self.midi_file:
            print("Error: No hay archivo MIDI cargado")
            return
        
        if not ruta_salida:
            if self.archivo_original:
                ruta_salida = self.archivo_original
            else:
                ruta_salida = "output.mid"
        
        # Crear directorio si no existe
        directorio = os.path.dirname(ruta_salida)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio)
        
        try:
            self.midi_file.save(ruta_salida)
            print(f"Archivo guardado exitosamente: {ruta_salida}")
        except Exception as e:
            print(f"Error al guardar el archivo: {e}")
    
    def exportar_a_texto(self, ruta_salida):
        """
        Exporta el contenido del archivo MIDI a un archivo de texto
        
        Args:
            ruta_salida (str): Ruta del archivo de texto de salida
        """
        if not self.midi_file:
            print("Error: No hay archivo MIDI cargado")
            return
        
        try:
            with open(ruta_salida, 'w', encoding='utf-8') as archivo:
                archivo.write(f"Información del archivo MIDI\n")
                archivo.write(f"{'='*40}\n")
                archivo.write(f"Tipo: {self.midi_file.type}\n")
                archivo.write(f"Pistas: {len(self.midi_file.tracks)}\n")
                archivo.write(f"Ticks por beat: {self.midi_file.ticks_per_beat}\n")
                archivo.write(f"Duración: {self.midi_file.length:.2f} segundos\n\n")
                
                for i, track in enumerate(self.midi_file.tracks):
                    archivo.write(f"Pista {i}:\n")
                    archivo.write(f"{'-'*20}\n")
                    for mensaje in track:
                        archivo.write(f"{mensaje}\n")
                    archivo.write("\n")
            
            print(f"Archivo de texto exportado: {ruta_salida}")
        except Exception as e:
            print(f"Error al exportar a texto: {e}")