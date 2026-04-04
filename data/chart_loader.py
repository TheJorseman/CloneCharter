import re
from typing import Dict, List, Any, Optional

class CloneHeroChartParser:
    def __init__(self, chart_text: str):
        # Eliminar BOM y otros caracteres especiales al inicio
        self.chart_text = self._clean_input_text(chart_text)
        self.sections: Dict[str, List[str]] = {}
        self.parsed_sections: Dict[str, Any] = {}
        self._parse_chart()
        self._parse_section_data()

    def _clean_input_text(self, text: str) -> str:
        """Limpia el texto de entrada eliminando caracteres especiales problemáticos"""
        # Eliminar BOM UTF-8 si existe
        if text.startswith('\ufeff'):
            text = text[1:]
        
        # Eliminar otros posibles BOMs
        if text.startswith('\ufffe'):
            text = text[1:]
        if text.startswith('\xef\xbb\xbf'):
            text = text[3:]
        
        # Eliminar espacios en blanco y caracteres de control al inicio
        text = text.lstrip('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')
        
        return text

    def _parse_chart(self):
        """Parse el archivo chart dividiéndolo en secciones"""
        current_section = None
        current_content = []
        
        for line in self.chart_text.splitlines():
            line = line.strip()
            
            # Detectar inicio de nueva sección
            if line.startswith('[') and line.endswith(']'):
                # Guardar sección anterior si existe
                if current_section is not None:
                    self.sections[current_section] = current_content
                
                # Iniciar nueva sección
                current_section = line[1:-1]
                current_content = []
            
            # Agregar contenido a la sección actual
            elif line and current_section is not None:
                current_content.append(line)
        
        # Guardar última sección
        if current_section is not None:
            self.sections[current_section] = current_content

    def _parse_section_data(self):
        """Parse el contenido de cada sección según su tipo"""
        for section_name, content in self.sections.items():
            if section_name == "Song":
                self.parsed_sections[section_name] = self._parse_song_section(content)
            elif section_name in ["SyncTrack", "Events"] or any(diff in section_name for diff in ["Expert", "Hard", "Medium", "Easy"]):
                self.parsed_sections[section_name] = self._parse_track_section(content)
            else:
                # Para secciones desconocidas, mantener como lista de strings
                self.parsed_sections[section_name] = content

    def _parse_song_section(self, content: List[str]) -> Dict[str, Any]:
        """Parse la sección Song que contiene metadata"""
        song_data = {}
        
        for line in content:
            if line in ['{', '}']:
                continue
                
            # Parse líneas como: Name = "El Rescate (ft. Junior H)"
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remover comillas si existen
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                # Convertir valores numéricos
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                
                song_data[key] = value
        
        return song_data

    def _parse_track_section(self, content: List[str]) -> List[Dict[str, Any]]:
        """Parse secciones de tracks (SyncTrack, Events, notas de instrumentos)"""
        track_data = []
        
        for line in content:
            if line in ['{', '}']:
                continue
                
            # Parse líneas como: 1536 = N 2 0 o 0 = TS 4
            if '=' in line:
                try:
                    position, event_data = line.split('=', 1)
                    position = int(position.strip())
                    event_data = event_data.strip()
                    
                    # Dividir datos del evento
                    event_parts = event_data.split()
                    
                    track_entry = {
                        'position': position,
                        'type': event_parts[0] if event_parts else '',
                        'data': event_parts[1:] if len(event_parts) > 1 else []
                    }
                    
                    track_data.append(track_entry)
                except (ValueError, IndexError):
                    # Si hay error parseando, mantener la línea raw
                    track_data.append({'raw_line': line})
        
        return track_data

    @classmethod
    def from_file(cls, file_path: str, encoding: str = 'utf-8-sig') -> 'CloneHeroChartParser':
        """Crea un parser directamente desde un archivo"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Intentar con diferentes encodings si falla
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"No se pudo leer el archivo {file_path} con ningún encoding conocido")
        
        return cls(content)

    def get_sections(self) -> List[str]:
        """Obtiene lista de todas las secciones disponibles"""
        return list(self.sections.keys())

    def get_section_content(self, section_name: str) -> Optional[List[str]]:
        """Obtiene el contenido raw de una sección específica"""
        return self.sections.get(section_name)

    def get_parsed_section(self, section_name: str) -> Optional[Any]:
        """Obtiene el contenido parseado de una sección específica"""
        return self.parsed_sections.get(section_name)

    def get_song_metadata(self) -> Optional[Dict[str, Any]]:
        """Obtiene los metadatos de la canción"""
        return self.get_parsed_section("Song")

    def get_sync_track(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene información de sincronización/tempo"""
        return self.get_parsed_section("SyncTrack")

    def get_events(self) -> Optional[List[Dict[str, Any]]]:
        """Obtiene eventos del chart (secciones, letras, etc.)"""
        return self.get_parsed_section("Events")

    def get_instrument_track_numeric(self, instrument: str, difficulty: str = "Expert") -> List[Dict[str, int]]:
        """
        Representación numérica pura para fácil transformación
        """
        track_name = f"{difficulty}{instrument}"
        notes = self.parsed_sections.get(track_name, [])
        
        # Mapeo de tipos a números
        type_map = {
            'N': 1,  # Nota normal
            'S': 2,  # Starpower
            'E': 3,  # Evento
            'B': 4,  # BPM change
            'TS': 5  # Time signature
        }
        
        numeric_notes = []
        for note in notes:
            if note['type'] == 'N' and len(note['data']) >= 2:
                numeric_note = {
                    'position': note['position'],
                    'type': type_map.get(note['type'], 0),
                    'button': int(note['data'][0]),
                    'duration': int(note['data'][1])
                }
                numeric_notes.append(numeric_note)
        
        return numeric_notes

    def get_instrument_track_human(self, instrument: str, difficulty: str = "Expert") -> List[Dict[str, Any]]:
        """
        Representación humana legible
        """
        track_name = f"{difficulty}{instrument}"
        notes = self.parsed_sections.get(track_name, [])
        
        button_map = {
            '0': 'Verde',
            '1': 'Rojo',
            '2': 'Amarillo',
            '3': 'Azul',
            '4': 'Naranja',
            '5': 'Forzar HOPO',
            '6': 'Tap note',
            '7': 'Nota abierta'
        }
        
        human_notes = []
        for note in notes:
            if note['type'] == 'N' and len(note['data']) >= 2:
                duration = int(note['data'][1])
                button = note['data'][0]
                
                human_note = {
                    'position': note['position'],
                    'type': 'Nota',
                    'button': button_map.get(button, f'Botón desconocido ({button})'),
                    'duration': duration,
                    'duration_description': 'Nota simple' if duration == 0 else f'Nota sostenida ({duration} ticks)'
                }
                human_notes.append(human_note)
        
        return human_notes

    def get_instrument_track_interchange(self, instrument: str, difficulty: str = "Expert") -> str:
        """
        Formato de intercambio con otros programas (solo números separados por comas)
        Formato: position,type,button,duration
        """
        track_name = f"{difficulty}{instrument}"
        notes = self.parsed_sections.get(track_name, [])
        
        interchange_lines = []
        for note in notes:
            if note['type'] == 'N' and len(note['data']) >= 2:
                position = note['position']
                type_num = 1  # N = 1
                button = int(note['data'][0])
                duration = int(note['data'][1])
                
                interchange_lines.append(f"{position},{type_num},{button},{duration}")
        
        return "\n".join(interchange_lines) if interchange_lines else ""

    def get_all_instrument_tracks_formats(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todos los tracks en los tres formatos
        """
        all_formats = {}
        
        for track_name in self.sections.keys():
            if any(diff in track_name for diff in ['Easy', 'Medium', 'Hard', 'Expert']):
                # Extraer instrumento y dificultad del nombre
                for diff in ['Expert', 'Hard', 'Medium', 'Easy']:
                    if track_name.startswith(diff):
                        instrument = track_name[len(diff):]
                        difficulty = diff
                        break
                
                all_formats[track_name] = {
                    'numeric': self.get_instrument_track_numeric(instrument, difficulty),
                    'human': self.get_instrument_track_human(instrument, difficulty),
                    'interchange': self.get_instrument_track_interchange(instrument, difficulty)
                }
        
        return all_formats

    def print_track_all_formats(self, instrument: str, difficulty: str = "Expert", max_notes: int = 5):
        """
        Imprime ejemplos de los tres formatos
        """
        track_name = f"{difficulty}{instrument}"
        
        print(f"\n=== FORMATOS PARA {track_name} ===")
        
        # Formato numérico
        numeric_data = self.get_instrument_track_numeric(instrument, difficulty)
        print(f"\n1. FORMATO NUMÉRICO (para transformaciones):")
        for i, note in enumerate(numeric_data[:max_notes]):
            print(f"   {note}")
        if len(numeric_data) > max_notes:
            print(f"   ... y {len(numeric_data) - max_notes} notas más")
        
        # Formato humano
        human_data = self.get_instrument_track_human(instrument, difficulty)
        print(f"\n2. FORMATO HUMANO (legible):")
        for i, note in enumerate(human_data[:max_notes]):
            print(f"   Position: {note['position']}, Type: {note['type']}, Button: {note['button']}, Duration: {note['duration_description']}")
        if len(human_data) > max_notes:
            print(f"   ... y {len(human_data) - max_notes} notas más")
        
        # Formato intercambio
        interchange_data = self.get_instrument_track_interchange(instrument, difficulty)
        print(f"\n3. FORMATO INTERCAMBIO (CSV):")
        lines = interchange_data.split('\n')
        for line in lines[:max_notes]:
            if line:
                print(f"   {line}")
        if len(lines) > max_notes:
            print(f"   ... y {len(lines) - max_notes} líneas más")

    def get_available_instruments(self) -> List[str]:
        """
        Extrae todos los instrumentos disponibles en el chart
        """
        instruments = set()
        difficulties = ['Easy', 'Medium', 'Hard', 'Expert']
        
        for section_name in self.sections.keys():
            # Buscar secciones que contengan dificultades
            for difficulty in difficulties:
                if section_name.startswith(difficulty):
                    instrument = section_name[len(difficulty):]
                    if instrument:  # Si hay algo después de la dificultad
                        instruments.add(instrument)
                    break
        
        return sorted(list(instruments))

    def get_available_difficulties(self) -> List[str]:
        """
        Extrae todas las dificultades disponibles en el chart
        """
        difficulties_found = set()
        difficulty_levels = ['Easy', 'Medium', 'Hard', 'Expert']
        
        for section_name in self.sections.keys():
            for difficulty in difficulty_levels:
                if section_name.startswith(difficulty):
                    difficulties_found.add(difficulty)
                    break
        
        # Ordenar por nivel de dificultad
        difficulty_order = ['Easy', 'Medium', 'Hard', 'Expert']
        return [diff for diff in difficulty_order if diff in difficulties_found]

    def get_instruments_by_difficulty(self) -> Dict[str, List[str]]:
        """
        Obtiene qué instrumentos están disponibles para cada dificultad
        """
        difficulty_instruments = {}
        difficulties = ['Easy', 'Medium', 'Hard', 'Expert']
        
        for difficulty in difficulties:
            instruments = []
            for section_name in self.sections.keys():
                if section_name.startswith(difficulty):
                    instrument = section_name[len(difficulty):]
                    if instrument:
                        instruments.append(instrument)
            
            if instruments:
                difficulty_instruments[difficulty] = sorted(instruments)
        
        return difficulty_instruments

    def get_difficulties_by_instrument(self) -> Dict[str, List[str]]:
        """
        Obtiene qué dificultades están disponibles para cada instrumento
        """
        instrument_difficulties = {}
        difficulties = ['Easy', 'Medium', 'Hard', 'Expert']
        
        # Primero obtener todos los instrumentos
        all_instruments = self.get_available_instruments()
        
        for instrument in all_instruments:
            available_diffs = []
            for difficulty in difficulties:
                section_name = f"{difficulty}{instrument}"
                if section_name in self.sections:
                    available_diffs.append(difficulty)
            
            if available_diffs:
                instrument_difficulties[instrument] = available_diffs
        
        return instrument_difficulties

    def get_chart_completeness(self) -> Dict[str, Any]:
        """
        Analiza qué tan completo está el chart (instrumentos y dificultades)
        """
        instruments = self.get_available_instruments()
        difficulties = self.get_available_difficulties()
        instruments_by_diff = self.get_instruments_by_difficulty()
        difficulties_by_inst = self.get_difficulties_by_instrument()
        
        # Mapeo de nombres de instrumentos a nombres legibles
        instrument_names = {
            'Single': 'Guitarra Principal',
            'DoubleBass': 'Bajo',
            'DoubleRhythm': 'Guitarra Rítmica',
            'GuitarCoop': 'Guitarra Cooperativa',
            'Keys': 'Teclados',
            'Drums': 'Batería',
            'ProDrums': 'Batería Pro',
            'Vocals': 'Vocales',
            'Guitar6Fret': 'Guitarra 6 Trastes',
            'Bass6Fret': 'Bajo 6 Trastes'
        }
        
        completeness = {
            'total_instruments': len(instruments),
            'total_difficulties': len(difficulties),
            'instruments': instruments,
            'difficulties': difficulties,
            'instruments_human_readable': [instrument_names.get(inst, inst) for inst in instruments],
            'instruments_by_difficulty': instruments_by_diff,
            'difficulties_by_instrument': difficulties_by_inst,
            'is_full_difficulty': len(difficulties) == 4,  # Easy, Medium, Hard, Expert
            'missing_difficulties': [diff for diff in ['Easy', 'Medium', 'Hard', 'Expert'] if diff not in difficulties]
        }
        
        return completeness

    def print_chart_summary(self):
        """
        Imprime un resumen completo del chart
        """
        song_info = self.get_song_metadata()
        completeness = self.get_chart_completeness()
        
        print("=" * 60)
        print("RESUMEN DEL CHART")
        print("=" * 60)
        
        if song_info:
            print(f"Canción: {song_info.get('Name', 'Desconocido')}")
            print(f"Artista: {song_info.get('Artist', 'Desconocido')}")
            print(f"Charter: {song_info.get('Charter', 'Desconocido')}")
            print(f"Año: {song_info.get('Year', 'Desconocido')}")
            print(f"Género: {song_info.get('Genre', 'Desconocido')}")
        
        print(f"\nInstrumentos disponibles ({len(completeness['instruments'])}):")
        for i, (inst, readable) in enumerate(zip(completeness['instruments'], completeness['instruments_human_readable'])):
            print(f"  {i+1}. {readable} ({inst})")
        
        print(f"\nDificultades disponibles ({len(completeness['difficulties'])}):")
        for i, diff in enumerate(completeness['difficulties']):
            print(f"  {i+1}. {diff}")
        
        if not completeness['is_full_difficulty']:
            print(f"\n⚠️  Dificultades faltantes: {', '.join(completeness['missing_difficulties'])}")
        else:
            print(f"\n✅ Chart completo - Todas las dificultades disponibles")
        
        print(f"\nDificultades por instrumento:")
        for instrument, diffs in completeness['difficulties_by_instrument'].items():
            readable_name = completeness['instruments_human_readable'][completeness['instruments'].index(instrument)]
            print(f"  {readable_name}: {', '.join(diffs)}")
        
        print(f"\nInstrumentos por dificultad:")
        for difficulty, instruments in completeness['instruments_by_difficulty'].items():
            readable_instruments = [completeness['instruments_human_readable'][completeness['instruments'].index(inst)] 
                                for inst in instruments]
            print(f"  {difficulty}: {', '.join(readable_instruments)}")

    def is_chart_playable_for_difficulty(self, difficulty: str) -> bool:
        """
        Verifica si el chart es jugable para una dificultad específica
        """
        available_difficulties = self.get_available_difficulties()
        return difficulty in available_difficulties

    def get_recommended_difficulty(self) -> str:
        """
        Sugiere una dificultad basada en lo que está disponible
        """
        difficulties = self.get_available_difficulties()
        
        # Orden de preferencia para principiantes
        preference_order = ['Easy', 'Medium', 'Hard', 'Expert']
        
        for preferred in preference_order:
            if preferred in difficulties:
                return preferred
        
        return difficulties[0] if difficulties else None

    def get_instruments_organized_by_category(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Obtiene todos los instrumentos organizados por categoría (guitarra, bajo, drums)
        y ordenados por dificultad de Expert a Easy
        """
        # Categorías de instrumentos basadas en Clone Hero
        instrument_categories = {
            'guitarra': {
                'Single': 'Guitarra Principal',
                'DoubleRhythm': 'Guitarra Rítmica', 
                'GuitarCoop': 'Guitarra Cooperativa',
                'Guitar6Fret': 'Guitarra 6 Trastes'
            },
            'bajo': {
                'DoubleBass': 'Bajo',
                'Bass6Fret': 'Bajo 6 Trastes'
            },
            'drums': {
                'Drums': 'Batería',
                'ProDrums': 'Batería Pro'
            },
            'otros': {
                'Keys': 'Teclados',
                'Vocals': 'Vocales'
            }
        }
        
        # Orden de dificultades de Expert a Easy
        difficulty_order = ['Expert', 'Hard', 'Medium', 'Easy']
        
        organized_instruments = {}
        
        for category, instruments in instrument_categories.items():
            organized_instruments[category] = {}
            
            for difficulty in difficulty_order:
                found_instruments = []
                
                for instrument_code, instrument_name in instruments.items():
                    section_name = f"{difficulty}{instrument_code}"
                    if section_name in self.sections:
                        found_instruments.append({
                            'code': instrument_code,
                            'name': instrument_name,
                            'section': section_name,
                            'difficulty': difficulty
                        })
                
                if found_instruments:
                    organized_instruments[category][difficulty] = found_instruments
        
        return organized_instruments

    def get_expert_to_easy_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Resumen simplificado: Expert a Easy, ordenado por guitarra, bajo, drums
        """
        organized = self.get_instruments_organized_by_category()
        difficulty_order = ['Expert', 'Hard', 'Medium', 'Easy']
        
        summary = {}
        
        for difficulty in difficulty_order:
            summary[difficulty] = []
            
            # Primero guitarras
            if 'guitarra' in organized and difficulty in organized['guitarra']:
                for instrument in organized['guitarra'][difficulty]:
                    summary[difficulty].append({
                        'category': 'guitarra',
                        'instrument': instrument['name'],
                        'code': instrument['code'],
                        'section': instrument['section']
                    })
            
            # Luego bajos
            if 'bajo' in organized and difficulty in organized['bajo']:
                for instrument in organized['bajo'][difficulty]:
                    summary[difficulty].append({
                        'category': 'bajo',
                        'instrument': instrument['name'],
                        'code': instrument['code'],
                        'section': instrument['section']
                    })
            
            # Finalmente drums
            if 'drums' in organized and difficulty in organized['drums']:
                for instrument in organized['drums'][difficulty]:
                    summary[difficulty].append({
                        'category': 'drums',
                        'instrument': instrument['name'],
                        'code': instrument['code'],
                        'section': instrument['section']
                    })
            
            # Otros instrumentos al final
            if 'otros' in organized and difficulty in organized['otros']:
                for instrument in organized['otros'][difficulty]:
                    summary[difficulty].append({
                        'category': 'otros',
                        'instrument': instrument['name'],
                        'code': instrument['code'],
                        'section': instrument['section']
                    })
        
        return summary

    def print_instruments_by_category_and_difficulty(self):
        """
        Imprime todos los instrumentos organizados por categoría y dificultad
        """
        summary = self.get_expert_to_easy_summary()
        song_info = self.get_song_metadata()
        
        print("=" * 80)
        print("INSTRUMENTOS POR CATEGORÍA Y DIFICULTAD")
        print("=" * 80)
        
        if song_info:
            print(f"Chart: {song_info.get('Name', 'Desconocido')} - {song_info.get('Artist', 'Desconocido')}")
            print()
        
        category_icons = {
            'guitarra': '🎸',
            'bajo': '🎵',
            'drums': '🥁',
            'otros': '🎹'
        }
        
        for difficulty, instruments in summary.items():
            if instruments:  # Solo mostrar si hay instrumentos
                print(f"📋 **{difficulty.upper()}**")
                
                current_category = None
                for instrument in instruments:
                    if instrument['category'] != current_category:
                        current_category = instrument['category']
                        icon = category_icons.get(current_category, '🎵')
                        print(f"   {icon} {current_category.title()}:")
                    
                    print(f"     • {instrument['instrument']} ({instrument['code']})")
                
                print()  # Espacio entre dificultades

    def get_playable_combinations(self) -> List[Dict[str, Any]]:
        """
        Obtiene todas las combinaciones instrumento-dificultad disponibles
        ordenadas por categoría y dificultad
        """
        summary = self.get_expert_to_easy_summary()
        combinations = []
        
        for difficulty, instruments in summary.items():
            for instrument in instruments:
                combinations.append({
                    'difficulty': difficulty,
                    'category': instrument['category'],
                    'instrument': instrument['instrument'],
                    'code': instrument['code'],
                    'section': instrument['section'],
                    'recommended_for_beginners': difficulty in ['Easy', 'Medium'] and instrument['category'] in ['guitarra', 'bajo']
                })
        
        return combinations

    def get_category_completeness(self) -> Dict[str, Dict[str, Any]]:
        """
        Analiza qué tan completa está cada categoría de instrumentos
        """
        organized = self.get_instruments_organized_by_category()
        difficulty_order = ['Expert', 'Hard', 'Medium', 'Easy']
        
        completeness = {}
        
        for category, difficulties in organized.items():
            if not difficulties:  # Si no hay instrumentos en esta categoría
                continue
                
            total_possible = len(difficulty_order)
            available_difficulties = list(difficulties.keys())
            missing_difficulties = [d for d in difficulty_order if d not in available_difficulties]
            
            # Contar instrumentos únicos en la categoría
            unique_instruments = set()
            for difficulty_instruments in difficulties.values():
                for instrument in difficulty_instruments:
                    unique_instruments.add(instrument['code'])
            
            completeness[category] = {
                'total_instruments': len(unique_instruments),
                'available_difficulties': available_difficulties,
                'missing_difficulties': missing_difficulties,
                'completeness_percentage': (len(available_difficulties) / total_possible) * 100,
                'is_complete': len(missing_difficulties) == 0,
                'recommended_difficulty': available_difficulties[0] if available_difficulties else None
            }
        
        return completeness


    def __str__(self) -> str:
        """Representación string del parser"""
        song_data = self.get_song_metadata()
        if song_data:
            return f"Chart: {song_data.get('Name', 'Unknown')} by {song_data.get('Artist', 'Unknown Artist')}"
        return "Clone Hero Chart Parser"

    def __repr__(self) -> str:
        return f"CloneHeroChartParser(sections={len(self.sections)})"
