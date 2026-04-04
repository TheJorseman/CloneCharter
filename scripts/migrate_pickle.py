#!/usr/bin/env python3
"""
Script para procesar archivos pickle grandes con chunks de 500MB-5GB.
SIN subdivisión automática para evitar recursión infinita.
"""

import os
import gc
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def split_pickle_physical(pickle_path: str, output_dir: str, chunk_size_mb: int = 500) -> List[str]:
    """
    Divide físicamente el archivo pickle en chunks sin subdivisión.
    
    Args:
        pickle_path: Ruta al archivo pickle original
        output_dir: Directorio donde guardar los chunks
        chunk_size_mb: Tamaño de cada chunk en MB (500MB - 5GB)
    
    Returns:
        Lista de rutas a los archivos chunk creados
    """
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    file_size = os.path.getsize(pickle_path)
    total_chunks = (file_size + chunk_size_bytes - 1) // chunk_size_bytes
    
    print(f"📁 Dividiendo archivo de {file_size / 1024**3:.2f} GB en {total_chunks} chunks de {chunk_size_mb} MB")
    
    chunk_files = []
    
    with open(pickle_path, 'rb') as input_file:
        with tqdm(total=total_chunks, desc="🔄 Creando chunks") as pbar:
            for chunk_num in range(total_chunks):
                chunk_filename = output_path / f"chunk_{chunk_num:04d}.pkl"
                
                with open(chunk_filename, 'wb') as chunk_file:
                    remaining_bytes = file_size - input_file.tell()
                    bytes_to_read = min(chunk_size_bytes, remaining_bytes)
                    
                    if bytes_to_read <= 0:
                        break
                    
                    # Leer por bloques para no sobrecargar memoria
                    block_size = 64 * 1024 * 1024  # 64MB blocks
                    total_read = 0
                    
                    while total_read < bytes_to_read:
                        current_block_size = min(block_size, bytes_to_read - total_read)
                        data = input_file.read(current_block_size)
                        
                        if not data:
                            break
                            
                        chunk_file.write(data)
                        total_read += len(data)
                
                chunk_files.append(str(chunk_filename))
                pbar.update(1)
    
    print(f"✅ {len(chunk_files)} chunks creados exitosamente")
    return chunk_files

def try_load_chunk_directly(chunk_path: str) -> Optional[Any]:
    """
    Intenta cargar un chunk directamente SIN subdivisión.
    Si falla, simplemente retorna None.
    
    Args:
        chunk_path: Ruta al archivo chunk
        
    Returns:
        Datos del chunk o None si no se puede cargar
    """
    file_size = os.path.getsize(chunk_path)
    print(f"🔄 Intentando cargar chunk: {os.path.basename(chunk_path)} ({file_size / 1024**2:.1f} MB)")
    
    try:
        import pickle
        with open(chunk_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Chunk cargado exitosamente")
        return data
    except MemoryError:
        print(f"❌ Memoria insuficiente para chunk {os.path.basename(chunk_path)}")
        return None
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"❌ Error de pickle en chunk {os.path.basename(chunk_path)}: {type(e).__name__}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado en chunk {os.path.basename(chunk_path)}: {e}")
        return None

def extract_dataset_from_chunk(chunk_data: Any, chunk_id: int) -> Optional[Dataset]:
    """
    Extrae datos de dataset desde un chunk sin subdivisiones.
    
    Args:
        chunk_data: Datos del chunk
        chunk_id: ID del chunk
        
    Returns:
        Dataset de Hugging Face o None
    """
    try:
        # Caso 1: El chunk contiene la estructura completa del checkpoint
        if isinstance(chunk_data, dict) and 'all_data' in chunk_data:
            all_data = chunk_data['all_data']
            if all_data and isinstance(all_data, dict):
                # Verificar que tiene estructura de dataset (listas)
                has_lists = any(isinstance(v, list) for v in all_data.values())
                if has_lists:
                    print(f"✅ Dataset encontrado en chunk {chunk_id} - Estructura: {list(all_data.keys())}")
                    return Dataset.from_dict(all_data)
        
        # Caso 2: El chunk es directamente all_data
        elif isinstance(chunk_data, dict) and chunk_data:
            # Verificar si parece ser datos de dataset
            has_lists = any(isinstance(v, list) for v in chunk_data.values())
            if has_lists:
                print(f"✅ Datos directos encontrados en chunk {chunk_id} - Claves: {list(chunk_data.keys())}")
                return Dataset.from_dict(chunk_data)
        
        # Caso 3: Datos no válidos
        print(f"⚠️ Chunk {chunk_id} no contiene datos de dataset válidos - Tipo: {type(chunk_data)}")
        if isinstance(chunk_data, dict):
            print(f"   Claves disponibles: {list(chunk_data.keys())}")
        
        return None
            
    except Exception as e:
        print(f"❌ Error extrayendo dataset del chunk {chunk_id}: {e}")
        return None

def process_large_chunks(chunk_files: List[str]) -> List[Dataset]:
    """
    Procesa chunks grandes sin subdivisión.
    
    Args:
        chunk_files: Lista de archivos chunk
        
    Returns:
        Lista de datasets válidos
    """
    datasets = []
    successful_chunks = 0
    
    print(f"\n🔄 PROCESANDO {len(chunk_files)} CHUNKS (SIN SUBDIVISIÓN)")
    print("-" * 60)
    
    with tqdm(total=len(chunk_files), desc="🔄 Procesando chunks") as pbar:
        for i, chunk_file in enumerate(chunk_files):
            print(f"\n--- Chunk {i+1}/{len(chunk_files)} ---")
            
            # Cargar chunk directamente
            chunk_data = try_load_chunk_directly(chunk_file)
            
            if chunk_data is None:
                print(f"⚠️ Saltando chunk {i+1} (no se pudo cargar)")
                pbar.update(1)
                continue
            
            # Extraer dataset
            dataset = extract_dataset_from_chunk(chunk_data, i+1)
            
            if dataset is not None:
                datasets.append(dataset)
                successful_chunks += 1
                print(f"✅ Chunk {i+1}: {len(dataset)} elementos extraídos")
            else:
                print(f"⚠️ Chunk {i+1}: sin datos de dataset válidos")
            
            # Limpiar memoria después de cada chunk
            del chunk_data
            if dataset is not None:
                # No eliminar dataset aquí, lo necesitamos para combinar
                pass
            gc.collect()
            
            pbar.update(1)
    
    print(f"\n📊 Resumen: {successful_chunks}/{len(chunk_files)} chunks procesados exitosamente")
    return datasets

def combine_datasets_safely(datasets: List[Dataset]) -> Dataset:
    """
    Combina datasets de forma segura.
    
    Args:
        datasets: Lista de datasets a combinar
        
    Returns:
        Dataset combinado
    """
    if not datasets:
        raise ValueError("No hay datasets para combinar")
    
    if len(datasets) == 1:
        return datasets[0]
    
    print(f"\n🔗 COMBINANDO {len(datasets)} DATASETS")
    print("-" * 40)
    
    # Mostrar información de cada dataset
    total_elements = 0
    for i, ds in enumerate(datasets):
        elements = len(ds)
        total_elements += elements
        print(f"Dataset {i+1}: {elements:,} elementos")
    
    print(f"Total esperado: {total_elements:,} elementos")
    
    # Combinar progresivamente
    combined = datasets[0]
    
    with tqdm(total=len(datasets)-1, desc="🔗 Combinando") as pbar:
        for i, dataset in enumerate(datasets[1:], 1):
            try:
                combined = concatenate_datasets([combined, dataset])
                print(f"✅ Combinado dataset {i+1} - Total: {len(combined):,} elementos")
            except Exception as e:
                print(f"❌ Error combinando dataset {i+1}: {e}")
                continue
            
            # Limpiar dataset individual para liberar memoria
            del dataset
            gc.collect()
            
            pbar.update(1)
    
    print(f"🎉 Combinación completada: {len(combined):,} elementos finales")
    return combined

def process_large_pickle(pickle_path: str, output_dir: str, 
                        chunk_size_mb: int = 500,
                        cleanup_chunks: bool = True) -> Optional[str]:
    """
    Procesa archivo pickle con chunks grandes sin subdivisión.
    
    Args:
        pickle_path: Ruta al archivo pickle original
        output_dir: Directorio de salida
        chunk_size_mb: Tamaño de chunks en MB (500-5000)
        cleanup_chunks: Si eliminar chunks temporales
        
    Returns:
        Ruta al dataset final o None si falla
    """
    
    print("🚀 PROCESADOR DE PICKLE CON CHUNKS GRANDES")
    print("=" * 60)
    print(f"📁 Archivo: {pickle_path}")
    print(f"📏 Tamaño: {os.path.getsize(pickle_path) / 1024**3:.2f} GB")
    print(f"📦 Chunk size: {chunk_size_mb} MB ({chunk_size_mb/1024:.1f} GB)")
    print(f"🔄 Sin subdivisión automática")
    print("=" * 60)
    
    # Validar tamaño de chunk
    if chunk_size_mb < 100 or chunk_size_mb > 5000:
        print(f"⚠️ Advertencia: Chunk size de {chunk_size_mb} MB puede no ser óptimo")
        print(f"   Recomendado: 500-5000 MB")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Crear directorio temporal para chunks
    temp_chunks_dir = output_path / "temp_chunks"
    temp_chunks_dir.mkdir(exist_ok=True)
    
    try:
        # Paso 1: Dividir archivo en chunks grandes
        print(f"\n📦 PASO 1: CREANDO CHUNKS DE {chunk_size_mb} MB")
        print("-" * 50)
        
        chunk_files = split_pickle_physical(pickle_path, str(temp_chunks_dir), chunk_size_mb)
        
        if not chunk_files:
            print("❌ No se pudieron crear chunks")
            return None
        
        # Paso 2: Procesar chunks (sin subdivisión)
        datasets = process_large_chunks(chunk_files)
        
        if not datasets:
            print("❌ No se procesaron chunks exitosamente")
            return None
        
        # Paso 3: Combinar datasets
        print(f"\n🔗 PASO 3: COMBINANDO DATASETS")
        print("-" * 50)
        
        final_dataset = combine_datasets_safely(datasets)
        
        # Paso 4: Guardar dataset final
        print(f"\n💾 PASO 4: GUARDANDO DATASET FINAL")
        print("-" * 50)
        
        final_path = output_path / "final_dataset"
        final_dataset.save_to_disk(str(final_path))
        
        total_elements = len(final_dataset)
        print(f"✅ Dataset guardado: {final_path}")
        print(f"📊 Total elementos: {total_elements:,}")
        
        # Paso 5: Guardar metadata
        metadata = {
            'processing_info': {
                'method': 'large_chunks_no_subdivision',
                'original_file': pickle_path,
                'original_size_gb': os.path.getsize(pickle_path) / 1024**3,
                'processed_at': datetime.now().isoformat(),
                'chunk_size_mb': chunk_size_mb,
                'total_chunks': len(chunk_files),
                'successful_chunks': len(datasets),
                'total_elements': total_elements
            }
        }
        
        metadata_path = output_path / "processing_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Paso 6: Limpieza opcional
        if cleanup_chunks:
            print(f"\n🧹 PASO 6: LIMPIANDO CHUNKS TEMPORALES")
            print("-" * 50)
            
            shutil.rmtree(temp_chunks_dir)
            print("✅ Chunks temporales eliminados")
        else:
            print(f"\n📁 Chunks conservados en: {temp_chunks_dir}")
        
        # Resultado final
        print("\n" + "=" * 60)
        print("🎉 PROCESAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"📁 Dataset final: {final_path}")
        print(f"📊 Elementos: {total_elements:,}")
        print(f"📦 Chunks procesados: {len(datasets)}/{len(chunk_files)}")
        
        return str(final_path)
        
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        return None
    
    finally:
        gc.collect()

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Procesa archivos pickle con chunks grandes (500MB-5GB) SIN subdivisión",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Chunks de 500MB (recomendado)
python script.py -i checkpoint.pkl -o output/ -s 500

# Chunks de 1GB 
python script.py -i checkpoint.pkl -o output/ -s 1000

# Chunks de 2GB
python script.py -i checkpoint.pkl -o output/ -s 2000

# Chunks de 5GB (máximo recomendado)
python script.py -i checkpoint.pkl -o output/ -s 5000

Características:
- SIN subdivisión automática (evita recursión infinita)
- Chunks de 500MB a 5GB
- Procesamiento directo y simple
- Control de memoria mejorado
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Archivo pickle de entrada"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directorio de salida"
    )
    parser.add_argument(
        "--chunk-size", "-s",
        type=int,
        default=500,
        help="Tamaño de chunks en MB (100-5000, recomendado: 500-2000)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="No eliminar chunks temporales"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Información detallada"
    )
    
    args = parser.parse_args()
    
    # Validaciones
    if not os.path.exists(args.input):
        print(f"❌ Error: El archivo {args.input} no existe")
        return 1
    
    if args.chunk_size < 100 or args.chunk_size > 5000:
        print(f"❌ Error: Chunk size debe estar entre 100 y 5000 MB")
        return 1
    
    if args.verbose:
        print(f"Configuración:")
        print(f"  Archivo: {args.input}")
        print(f"  Salida: {args.output}")
        print(f"  Chunk size: {args.chunk_size} MB")
        print(f"  Sin subdivisión: ✅")
        print(f"  Limpiar chunks: {not args.no_cleanup}")
        print()
    
    # Procesar
    result = process_large_pickle(
        pickle_path=args.input,
        output_dir=args.output,
        chunk_size_mb=args.chunk_size,
        cleanup_chunks=not args.no_cleanup
    )
    
    if result:
        print(f"\n🎉 ¡Éxito! Dataset en: {result}")
        return 0
    else:
        print(f"\n💥 Procesamiento falló")
        return 1

if __name__ == "__main__":
    exit(main())

#python migrate_pickle.py -i checkpoints/dataset_checkpoint.pkl -o chart_clone_hero_dataset/ -s 2000 --no-cleanup