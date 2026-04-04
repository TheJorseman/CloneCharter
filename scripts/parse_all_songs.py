import os
import json
import time
from data.chart_loader import ChartParser

def scan_folders_for_charts_and_ogg(root_path):
    folders_with_chart = []
    folders_without_chart = []
    all_ogg_files = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        has_chart = any(f.endswith('.chart') for f in filenames)
        if has_chart:
            folders_with_chart.append(dirpath)
        else:
            if filenames:
                folders_without_chart.append(dirpath)

        for filename in filenames:
            if filename.endswith('.ogg'):
                all_ogg_files.append(os.path.join(dirpath, filename))

    return folders_with_chart, folders_without_chart, all_ogg_files

def parse_chart_and_collect_ogg(folders_with_chart):
    parser = ChartParser()
    songs = []

    for folder in folders_with_chart:
        chart_file = None
        ogg_files = []
        
        # Buscar archivos .chart y .ogg en la carpeta
        for filename in os.listdir(folder):
            if filename.endswith('.chart'):
                chart_file = os.path.join(folder, filename)
            if filename.endswith('.ogg'):
                ogg_files.append(filename)

        if chart_file:
            try:
                parser.parse(chart_file)
                sections = parser.get_all_sections()
                folder_sections = {}
                
                for section in sections:
                    folder_sections[section] = parser.get_section(section)

                # Estructura modificada según tu especificación
                song_entry = {
                    "carpeta": folder,
                    "sections": folder_sections,
                    "ogg_files": ogg_files
                }
                songs.append(song_entry)
                
            except Exception as e:
                # En caso de error, también mantener la estructura
                song_entry = {
                    "carpeta": folder,
                    "sections": {"error": str(e)},
                    "ogg_files": ogg_files
                }
                songs.append(song_entry)

    return songs

def save_json_progressive(data, filename):
    """Guarda el JSON de forma gradual para evitar pérdida de datos"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Forzar escritura al disco
        print(f"Archivo JSON guardado exitosamente: {filename}")
        return True
    except Exception as e:
        print(f"Error guardando JSON: {e}")
        return False

def main(root_path, output_filename='chart_analysis.json'):
    print(f"Iniciando análisis de carpeta: {root_path}")
    
    # Escanear carpetas
    folders_with_chart, folders_without_chart, all_ogg_files = scan_folders_for_charts_and_ogg(root_path)
    
    # Parsear archivos .chart y recopilar archivos .ogg
    songs = parse_chart_and_collect_ogg(folders_with_chart)

    # Crear estructura JSON final
    json_data = {
        'root_path': root_path,
        'scan_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_folders_with_chart': len(folders_with_chart),
            'total_folders_without_chart': len(folders_without_chart),
            'total_ogg_files': len(all_ogg_files)
        },
        'folders_with_chart': folders_with_chart,
        'folders_without_chart': folders_without_chart,
        'songs': songs
    }

    # Guardar JSON de forma gradual
    save_json_progressive(json_data, output_filename)
    
    return json_data

# Ejemplo de uso
if __name__ == "__main__":
    root_directory = "G:\Songs"  # Cambiar por la ruta real
    result = main(root_directory, 'chart_analysis_results.json')
    
    print("\n=== RESUMEN ===")
    print(f"Carpeta raíz: {result['root_path']}")
    print(f"Canciones encontradas: {len(result['songs'])}")
    print(f"Carpetas sin .chart: {result['summary']['total_folders_without_chart']}")
