import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict
import numpy as np
import ast

def load_json_data(filename):
    """Carga los datos del archivo JSON"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo {filename} no es un JSON válido")
        return None

def extract_section_analysis(json_data):
    """Extrae análisis de secciones del JSON"""
    # Contar cuántas canciones tienen cada sección
    section_counts = defaultdict(int)
    
    # Recopilar longitudes de cada sección
    section_lengths = defaultdict(list)
    
    for song in json_data['songs']:
        for section, content in song['sections'].items():
            section_counts[section] += 1
            section_lengths[section].append(len(str(content)))
    
    return dict(section_counts), dict(section_lengths)

def extract_song_data(json_data):
    """Extrae datos numéricos de la sección Song"""
    song_numeric_data = defaultdict(list)
    
    for song in json_data['songs']:
        if 'Song' in song['sections']:
            song_content = song['sections']['Song']
            
            # Si el contenido es un diccionario, procesarlo directamente
            if isinstance(song_content, dict):
                for key, value in song_content.items():
                    try:
                        # Limpiar y convertir valores
                        if isinstance(value, str):
                            clean_value = value.replace('"', '').replace(',', '').strip()
                            numeric_value = float(clean_value)
                        else:
                            numeric_value = float(value)
                        song_numeric_data[key].append(numeric_value)
                    except (ValueError, TypeError):
                        pass
            else:
                # Si es texto, parsearlo línea por línea
                lines = str(song_content).split('\n')
                for line in lines:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            clean_value = value.replace('"', '').replace(',', '').strip()
                            numeric_value = float(clean_value)
                            song_numeric_data[key].append(numeric_value)
                        except (ValueError, TypeError):
                            pass
    
    return dict(song_numeric_data)

def extract_synctrack_data_fixed(json_data):
    """Extrae datos de SyncTrack para analizar valores B y TS - VERSIÓN CORREGIDA"""
    synctrack_counts = {"B": 0, "TS": 0}
    synctrack_values = {"B": [], "TS": []}
    
    for song in json_data['songs']:
        if 'SyncTrack' in song['sections']:
            synctrack_content = song['sections']['SyncTrack']
            
            # Si el contenido es un diccionario, procesarlo directamente
            if isinstance(synctrack_content, dict):
                for position, data in synctrack_content.items():
                    if isinstance(data, list) and len(data) >= 2:
                        sync_type = data[0]
                        sync_value = data[1]
                        
                        if sync_type in ['B', 'TS']:
                            synctrack_counts[sync_type] += 1
                            try:
                                numeric_value = float(sync_value)
                                synctrack_values[sync_type].append(numeric_value)
                            except (ValueError, TypeError):
                                pass
            else:
                # Intentar parsear como string con formato especial
                try:
                    # Intentar evaluar como diccionario literal
                    synctrack_dict = ast.literal_eval(str(synctrack_content))
                    for position, data in synctrack_dict.items():
                        if isinstance(data, list) and len(data) >= 2:
                            sync_type = data[0]
                            sync_value = data[1]
                            
                            if sync_type in ['B', 'TS']:
                                synctrack_counts[sync_type] += 1
                                try:
                                    numeric_value = float(sync_value)
                                    synctrack_values[sync_type].append(numeric_value)
                                except (ValueError, TypeError):
                                    pass
                except:
                    # Si falla, intentar parsing línea por línea
                    lines = str(synctrack_content).split('\n')
                    for line in lines:
                        line = line.strip()
                        if '=' in line:
                            try:
                                parts = line.split('=', 1)
                                if len(parts) >= 2:
                                    right_side = parts[1].strip()
                                    # Buscar patrones como "B 120000" o "TS 4"
                                    if ' ' in right_side:
                                        elements = right_side.split()
                                        if len(elements) >= 2:
                                            sync_type = elements[0].strip()
                                            sync_value = elements[1].strip()
                                            
                                            if sync_type in ['B', 'TS']:
                                                synctrack_counts[sync_type] += 1
                                                try:
                                                    numeric_value = float(sync_value)
                                                    synctrack_values[sync_type].append(numeric_value)
                                                except (ValueError, TypeError):
                                                    pass
                            except Exception:
                                continue
    
    return synctrack_counts, synctrack_values

def extract_basic_analysis_data(json_data):
    """Extrae datos básicos para análisis original"""
    # Número de sections por canción
    num_sections = [len(song['sections']) for song in json_data['songs']]
    
    # Número de archivos ogg por canción
    num_ogg_files = [len(song['ogg_files']) for song in json_data['songs']]
    
    # Longitud de las secciones (caracteres en el contenido)
    section_lengths = []
    for song in json_data['songs']:
        for section_content in song['sections'].values():
            section_lengths.append(len(str(section_content)))
    
    return {
        'num_sections': num_sections,
        'num_ogg_files': num_ogg_files,
        'section_lengths': section_lengths
    }

def create_comprehensive_analysis(json_data):
    """Crea análisis completo con todos los histogramas"""
    
    # Extraer datos
    section_counts, section_lengths = extract_section_analysis(json_data)
    song_data = extract_song_data(json_data)
    synctrack_counts, synctrack_values = extract_synctrack_data_fixed(json_data)
    
    # Crear figura con 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Histograma de número de canciones por sección',
            'Histograma de longitud promedio de cada sección',
            'Histograma de datos numéricos de Song',
            'Histograma de valores B y TS en SyncTrack'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Histograma número de canciones por sección
    fig.add_trace(
        go.Bar(
            x=list(section_counts.keys()),
            y=list(section_counts.values()),
            name="Canciones por sección",
            marker_color='#2E8B8B'
        ),
        row=1, col=1
    )

    # 2. Histograma longitud promedio de cada sección
    avg_lengths = {k: np.mean(v) for k, v in section_lengths.items()}
    fig.add_trace(
        go.Bar(
            x=list(avg_lengths.keys()),
            y=list(avg_lengths.values()),
            name="Longitud promedio de sección",
            marker_color='#2E8B8B'
        ),
        row=1, col=2
    )

    # 3. Histograma datos numéricos de Song
    colors = ['#B85450', '#C53030', '#4A90E2', '#50C878', '#FFD700']
    song_keys = list(song_data.keys())
    song_values = [np.mean(values) if values else 0 for values in song_data.values()]
    
    if song_keys and song_values:
        fig.add_trace(
            go.Bar(
                x=song_keys,
                y=song_values,
                name="Datos Song",
                marker_color=colors[:len(song_keys)]
            ),
            row=2, col=1
        )

    # 4. Histograma valores B y TS en SyncTrack
    fig.add_trace(
        go.Bar(
            x=list(synctrack_counts.keys()),
            y=list(synctrack_counts.values()),
            name="SyncTrack B y TS",
            marker_color=['#2E8B8B', '#B85450']
        ),
        row=2, col=2
    )

    # Configurar layout
    fig.update_layout(
        height=700,
        width=1200,
        title_text='Análisis de datos extendido del archivo JSON',
        showlegend=False
    )

    # Actualizar títulos de ejes
    fig.update_xaxes(title_text="Secciones", row=1, col=1)
    fig.update_xaxes(title_text="Secciones", row=1, col=2)
    fig.update_xaxes(title_text="Parámetros", row=2, col=1)
    fig.update_xaxes(title_text="Tipo", row=2, col=2)
    
    fig.update_yaxes(title_text="Número de canciones", row=1, col=1)
    fig.update_yaxes(title_text="Longitud promedio", row=1, col=2)
    fig.update_yaxes(title_text="Valor promedio", row=2, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=2)

    return fig

def create_basic_analysis(json_data):
    """Crea análisis básico original"""
    
    basic_data = extract_basic_analysis_data(json_data)
    synctrack_counts, synctrack_values = extract_synctrack_data_fixed(json_data)
    
    # Crear figura con 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Histograma de número de sections por canción',
            'Histograma de número de archivos ogg por canción',
            'Histograma de longitud de las secciones',
            'Histograma de valores en SyncTrack'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Histograma número de sections
    fig.add_trace(
        go.Histogram(
            x=basic_data['num_sections'], 
            nbinsx=10, 
            name='Sections',
            marker_color='#2E8B8B'
        ),
        row=1, col=1
    )

    # Histograma número de archivos ogg
    fig.add_trace(
        go.Histogram(
            x=basic_data['num_ogg_files'], 
            nbinsx=10, 
            name='Ogg Files',
            marker_color='#2E8B8B'
        ),
        row=1, col=2
    )

    # Histograma longitud de secciones
    fig.add_trace(
        go.Histogram(
            x=basic_data['section_lengths'], 
            nbinsx=10, 
            name='Section Lengths',
            marker_color='#B85450'
        ),
        row=2, col=1
    )

    # Histograma valores SyncTrack
    all_synctrack_values = synctrack_values['B'] + synctrack_values['TS']
    if all_synctrack_values:
        fig.add_trace(
            go.Histogram(
                x=all_synctrack_values, 
                nbinsx=10, 
                name='SyncTrack Values',
                marker_color='#C53030'
            ),
            row=2, col=2
        )

    # Configurar layout
    fig.update_layout(
        height=700, 
        width=900,
        title_text='Análisis de datos del archivo JSON',
        showlegend=False,
        font=dict(size=12)
    )

    # Actualizar ejes
    fig.update_xaxes(title_text="Número de sections", row=1, col=1)
    fig.update_xaxes(title_text="Número de archivos ogg", row=1, col=2)
    fig.update_xaxes(title_text="Longitud (caracteres)", row=2, col=1)
    fig.update_xaxes(title_text="Valor", row=2, col=2)
    
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=2)

    return fig

def create_detailed_section_length_analysis(section_lengths):
    """Crea análisis detallado de longitudes por sección"""
    
    fig = go.Figure()
    
    # Crear boxplot para cada sección
    for section, lengths in section_lengths.items():
        fig.add_trace(
            go.Box(
                y=lengths,
                name=section,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            )
        )
    
    fig.update_layout(
        title='Distribución de longitudes por sección',
        yaxis_title='Longitud (caracteres)',
        xaxis_title='Secciones',
        height=500,
        width=1000
    )
    
    return fig

def create_synctrack_values_distribution(synctrack_values):
    """Crea histograma de distribución de valores en SyncTrack"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribución de valores B (BPM)', 'Distribución de valores TS (Time Signature)')
    )
    
    # Histograma para valores B
    if synctrack_values['B']:
        fig.add_trace(
            go.Histogram(
                x=synctrack_values['B'],
                name='Valores B',
                marker_color='#2E8B8B',
                nbinsx=1000
            ),
            row=1, col=1
        )
    
    # Histograma para valores TS
    if synctrack_values['TS']:
        fig.add_trace(
            go.Histogram(
                x=synctrack_values['TS'],
                name='Valores TS',
                marker_color='#B85450',
                nbinsx=10
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        width=1000,
        title_text='Distribución de valores en SyncTrack',
        showlegend=False
    )
    
    return fig

def print_detailed_statistics(section_counts, section_lengths, song_data, synctrack_counts, synctrack_values, basic_data):
    """Imprime estadísticas detalladas"""
    
    print("=== ANÁLISIS DETALLADO DE SECCIONES ===")
    print("\nConteo de secciones por canción:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count} canciones")
    
    print("\nLongitud promedio de secciones:")
    for section, lengths in section_lengths.items():
        avg_length = np.mean(lengths)
        print(f"  {section}: {avg_length:.1f} caracteres")
    
    print("\nDatos numéricos de Song:")
    for param, values in song_data.items():
        if values:
            print(f"  {param}: promedio {np.mean(values):.2f}")
    
    print("\nConteo de elementos en SyncTrack:")
    for sync_type, count in synctrack_counts.items():
        print(f"  {sync_type}: {count} ocurrencias")
        if synctrack_values[sync_type]:
            print(f"    Rango: {min(synctrack_values[sync_type]):.0f} - {max(synctrack_values[sync_type]):.0f}")
    
    print(f"\nEstadísticas básicas:")
    print(f"  Promedio de secciones por canción: {np.mean(basic_data['num_sections']):.1f}")
    print(f"  Promedio de archivos OGG por canción: {np.mean(basic_data['num_ogg_files']):.1f}")
    print(f"  Longitud promedio de secciones: {np.mean(basic_data['section_lengths']):.1f} caracteres")

def main(json_filename='chart_analysis_results.json'):
    """Función principal completa"""
    
    # Cargar datos
    json_data = load_json_data(json_filename)
    if json_data is None:
        return
    
    # Extraer todos los datos
    section_counts, section_lengths = extract_section_analysis(json_data)
    song_data = extract_song_data(json_data)
    synctrack_counts, synctrack_values = extract_synctrack_data_fixed(json_data)
    basic_data = extract_basic_analysis_data(json_data)
    
    # Imprimir estadísticas
    print_detailed_statistics(section_counts, section_lengths, song_data, synctrack_counts, synctrack_values, basic_data)
    
    # Crear gráficos
    print("\nGenerando análisis visual...")
    
    # 1. Análisis extendido
    fig_extended = create_comprehensive_analysis(json_data)
    fig_extended.show()
    fig_extended.write_html("analisis_extendido.html")
    
    # 2. Análisis básico
    fig_basic = create_basic_analysis(json_data)
    fig_basic.show()
    fig_basic.write_html("analisis_basico.html")
    
    # 3. Análisis detallado de longitudes
    fig_lengths = create_detailed_section_length_analysis(section_lengths)
    fig_lengths.show()
    fig_lengths.write_html("analisis_longitudes.html")
    
    # 4. Análisis de SyncTrack
    fig_sync = create_synctrack_values_distribution(synctrack_values)
    fig_sync.show()
    fig_sync.write_html("analisis_synctrack.html")
    
    print("\nArchivos generados:")
    print("- analisis_extendido.html")
    print("- analisis_basico.html")
    print("- analisis_longitudes.html")
    print("- analisis_synctrack.html")

if __name__ == "__main__":
    main('chart_analysis_results.json')
