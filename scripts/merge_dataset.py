#!/usr/bin/env python3
"""
Script para unificar múltiples datasets de Hugging Face y subirlos al Hub
"""

import os
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from huggingface_hub import HfApi, login
import pandas as pd
from typing import List, Optional

class DatasetMerger:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.dataset_folders = [
            "clone_hero_dataset",
            "clone_hero_dataset_1", 
            "clone_hero_dataset_2",
            "clone_hero_dataset_3"
        ]
    
    def find_dataset_folders(self) -> List[Path]:
        """Encuentra todos los folders de datasets que existen"""
        existing_folders = []
        for folder_name in self.dataset_folders:
            folder_path = self.base_path / folder_name
            if folder_path.exists() and folder_path.is_dir():
                existing_folders.append(folder_path)
                print(f"✓ Encontrado: {folder_path}")
            else:
                print(f"✗ No encontrado: {folder_path}")
        return existing_folders
    
    def load_datasets(self, folders: List[Path]) -> List[Dataset]:
        """Carga todos los datasets desde los folders"""
        datasets = []
        
        for folder in folders:
            print(f"\nCargando dataset desde: {folder}")
            try:
                # Intenta cargar como dataset de Hugging Face
                dataset = load_from_disk(str(folder))
                
                # Si es un DatasetDict, toma el split 'train' o el primero disponible
                if isinstance(dataset, DatasetDict):
                    if 'train' in dataset:
                        dataset = dataset['train']
                    else:
                        dataset = dataset[list(dataset.keys())[0]]
                
                datasets.append(dataset)
                print(f"  ✓ Cargado: {len(dataset)} ejemplos")
                
            except Exception as e:
                print(f"  ✗ Error cargando {folder}: {e}")
                
                # Intenta cargar archivos individuales (CSV, JSON, etc.)
                self._try_load_individual_files(folder, datasets)
        
        return datasets
    
    def _try_load_individual_files(self, folder: Path, datasets: List[Dataset]):
        """Intenta cargar archivos individuales si no es un dataset de HF"""
        files_found = False
        
        # Buscar archivos CSV
        csv_files = list(folder.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dataset = Dataset.from_pandas(df)
                    datasets.append(dataset)
                    print(f"  ✓ Cargado CSV: {csv_file.name} ({len(dataset)} ejemplos)")
                    files_found = True
                except Exception as e:
                    print(f"  ✗ Error cargando CSV {csv_file}: {e}")
        
        # Buscar archivos JSON
        json_files = list(folder.glob("*.json"))
        if json_files:
            for json_file in json_files:
                try:
                    dataset = Dataset.from_json(str(json_file))
                    datasets.append(dataset)
                    print(f"  ✓ Cargado JSON: {json_file.name} ({len(dataset)} ejemplos)")
                    files_found = True
                except Exception as e:
                    print(f"  ✗ Error cargando JSON {json_file}: {e}")
        
        if not files_found:
            print(f"  ✗ No se encontraron archivos compatibles en {folder}")
    
    def merge_datasets(self, datasets: List[Dataset]) -> Dataset:
        """Une todos los datasets en uno solo"""
        if not datasets:
            raise ValueError("No hay datasets para unir")
        
        print(f"\nUniendo {len(datasets)} datasets...")
        
        # Añadir columna de origen para rastrear de dónde viene cada ejemplo
        for i, dataset in enumerate(datasets):
            # Añadir columna con el índice del dataset de origen
            dataset = dataset.add_column("source_dataset", [i] * len(dataset))
            datasets[i] = dataset
        
        # Concatenar todos los datasets
        merged_dataset = concatenate_datasets(datasets)
        
        print(f"✓ Dataset unificado creado: {len(merged_dataset)} ejemplos totales")
        return merged_dataset
    
    def save_merged_dataset(self, dataset: Dataset, output_path: str = "clone_hero_dataset_final"):
        """Guarda el dataset unificado localmente"""
        output_path = Path(output_path)
        
        print(f"\nGuardando dataset en: {output_path}")
        dataset.save_to_disk(str(output_path))
        print(f"✓ Dataset guardado exitosamente")
        
        # También guardar en formato CSV para fácil inspección
        #csv_path = output_path.with_suffix('.csv')
        #dataset.to_csv(str(csv_path))
        #print(f"✓ También guardado como CSV: {csv_path}")
    
    def upload_to_hub(self, dataset: Dataset, repo_name: str, token: Optional[str] = None, private: bool = False):
        """Sube el dataset al Hub de Hugging Face"""
        if token:
            login(token=token)
        
        print(f"\nSubiendo dataset al Hub: {repo_name}")
        
        try:
            # Crear DatasetDict con split de entrenamiento
            dataset_dict = DatasetDict({
                "train": dataset
            })
            
            # Subir al Hub
            dataset_dict.push_to_hub(
                repo_id=repo_name,
                private=private
            )
            
            print(f"✓ Dataset subido exitosamente a: https://huggingface.co/datasets/{repo_name}")
            
        except Exception as e:
            print(f"✗ Error subiendo al Hub: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Unificar datasets de Clone Hero")
    parser.add_argument("--base-path", default=".", help="Ruta base donde están los folders de datasets")
    parser.add_argument("--output", default="clone_hero_dataset_final", help="Nombre del dataset final")
    parser.add_argument("--upload", action="store_true", help="Subir al Hub de Hugging Face")
    parser.add_argument("--repo-name", help="Nombre del repositorio en el Hub (ej: username/dataset-name)")
    parser.add_argument("--token", help="Token de Hugging Face")
    parser.add_argument("--private", action="store_true", help="Hacer el dataset privado en el Hub")
    
    args = parser.parse_args()
    
    # Inicializar el merger
    merger = DatasetMerger(args.base_path)
    
    # Encontrar folders de datasets
    folders = merger.find_dataset_folders()
    
    if not folders:
        print("❌ No se encontraron folders de datasets")
        return
    
    # Cargar datasets
    datasets = merger.load_datasets(folders)
    
    if not datasets:
        print("❌ No se pudieron cargar datasets")
        return
    
    # Unir datasets
    merged_dataset = merger.merge_datasets(datasets)
    
    # Mostrar información del dataset final
    print(f"\n📊 Información del dataset final:")
    print(f"  - Total de ejemplos: {len(merged_dataset)}")
    print(f"  - Columnas: {list(merged_dataset.column_names)}")
    print(f"  - Características: {merged_dataset.features}")
    
    # Guardar localmente
    merger.save_merged_dataset(merged_dataset, args.output)
    
    # Subir al Hub si se solicita
    if args.upload:
        if not args.repo_name:
            print("❌ Se requiere --repo-name para subir al Hub")
            return
        
        api = HfApi(token=args.token)
        api.upload_folder(
            folder_path=args.output,
            repo_id=args.repo_name,
            repo_type="dataset",
        )

if __name__ == "__main__":
    main()


# python merge_dataset.py --base-path . --output clone_hero_dataset_final --upload --repo-name username/clone_hero_dataset --token <HF_TOKEN>