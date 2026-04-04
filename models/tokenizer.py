import torch
from transformers import PreTrainedTokenizer
from typing import List, Optional, Union, Dict, Any
import json
import os

class CloneHeroTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        max_duration_minutes: int = 120,
        max_beats: int = 512,
        max_beatshifts: int = 32,
        **kwargs
    ):
        # Tokens especiales básicos
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        
        # Tokens adicionales especiales
        self.instrument_tokens = ['<Guitar>', '<Bass>', '<Drums>']
        self.difficulty_tokens = ['<Expert>', '<Hard>', '<Medium>', '<Easy>']
        self.special_tokens = ['<Special>']
        self.note_type_tokens = ['<Normal>']  # Token para nota normal
        
        # Parámetros de configuración
        self.max_duration_minutes = max_duration_minutes
        self.max_beats = max_beats
        self.max_beatshifts = max_beatshifts
        
        # Crear vocabulario
        self.vocab = self._create_vocabulary()
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Configurar tokens especiales para el tokenizer base
        additional_special_tokens = (
            self.instrument_tokens + 
            self.difficulty_tokens + 
            self.special_tokens +
            self.note_type_tokens
        )
        
        super().__init__(
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs
        )
    
    def _create_vocabulary(self) -> Dict[str, int]:
        """Crea el vocabulario completo del tokenizer con Normal token"""
        vocab = {}
        idx = 0
        
        # Tokens especiales básicos
        basic_special = [self.bos_token, self.eos_token, self.unk_token, self.pad_token]
        for token in basic_special:
            vocab[token] = idx
            idx += 1
        
        # Tokens de instrumentos, dificultad, especiales y tipo de nota
        for token in (self.instrument_tokens + self.difficulty_tokens + 
                     self.special_tokens + self.note_type_tokens):
            vocab[token] = idx
            idx += 1
        
        # Tokens temporales
        for i in range(0, self.max_duration_minutes + 1):
            vocab[f'<Minute_{i}>'] = idx
            idx += 1
        
        for i in range(0, self.max_beats + 1):
            vocab[f'<Beat_{i}>'] = idx
            idx += 1
        
        for i in range(0, self.max_beatshifts + 1):
            vocab[f'<Beatshift_{i}>'] = idx
            idx += 1
        
        # Tokens Pitch 0-7 (para Guitar y Bass)
        for i in range(8):
            vocab[f'<Pitch_{i}>'] = idx
            idx += 1
        
        # Tokens DrumsPitch 0-4 (para Drums)
        for i in range(5):
            vocab[f'<DrumsPitch_{i}>'] = idx
            idx += 1
        
        return vocab
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        tokens = text.strip().split()
        valid_tokens = []
        for token in tokens:
            if token in self.vocab:
                valid_tokens.append(token)
            else:
                valid_tokens.append(self.unk_token)
        return valid_tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Método principal que emula el comportamiento de tokenizers de Hugging Face
        """
        # Manejar entrada simple vs batch
        is_batch = isinstance(text, list)
        if not is_batch:
            text = [text]
        
        # Tokenizar cada texto
        all_token_ids = []
        for single_text in text:
            tokens = self._tokenize(single_text)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            
            # Agregar tokens especiales si se solicita
            if add_special_tokens:
                if not token_ids or token_ids[0] != self.vocab[self.bos_token]:
                    token_ids = [self.vocab[self.bos_token]] + token_ids
                if not token_ids or token_ids[-1] != self.vocab[self.eos_token]:
                    token_ids = token_ids + [self.vocab[self.eos_token]]
            
            all_token_ids.append(token_ids)
        
        # Determinar longitud máxima para padding
        if padding == "longest" or (padding is True and max_length is None):
            max_len = max(len(ids) for ids in all_token_ids)
        elif max_length is not None:
            max_len = max_length
        else:
            max_len = None
        
        # Aplicar truncation y padding
        processed_ids = []
        attention_masks = []
        
        for token_ids in all_token_ids:
            # Truncation
            if truncation and max_len is not None and len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
                # Asegurar que termine con EOS si se agrega automáticamente
                if add_special_tokens:
                    token_ids[-1] = self.vocab[self.eos_token]
            
            # Padding
            if max_len is not None and len(token_ids) < max_len:
                pad_len = max_len - len(token_ids)
                token_ids = token_ids + [self.vocab[self.pad_token]] * pad_len
            
            # Crear attention mask
            attention_mask = [1 if token_id != self.vocab[self.pad_token] else 0 
                            for token_id in token_ids]
            
            processed_ids.append(token_ids)
            attention_masks.append(attention_mask)
        
        # Preparar output
        output = {
            'input_ids': processed_ids if is_batch else processed_ids[0],
            'attention_mask': attention_masks if is_batch else attention_masks[0]
        }
        
        # Convertir a tensors según el formato solicitado
        if return_tensors == 'pt':
            import torch
            output = {k: torch.tensor(v if is_batch else [v]) for k, v in output.items()}
        elif return_tensors == 'tf':
            try:
                import tensorflow as tf
                output = {k: tf.constant(v if is_batch else [v]) for k, v in output.items()}
            except ImportError:
                raise ImportError("TensorFlow no está instalado")
        elif return_tensors == 'np':
            import numpy as np
            output = {k: np.array(v if is_batch else [v]) for k, v in output.items()}
        
        return output
    
    def encode_complete_chart(
        self, 
        instrument: str, 
        difficulty: str, 
        beat_sequence: List[tuple]
    ) -> List[int]:
        """
        Codifica una secuencia de chart con la estructura:
        <BOS> <Instrument> <Difficulty> <Beatshift_X> <Type> <Pitch_Y> <Minute_Z> <Beat_W> <Beatshift_X>
        
        Args:
            instrument: Instrumento ('<Guitar>', '<Bass>', '<Drums>')
            difficulty: Dificultad ('<Expert>', '<Hard>', '<Medium>', '<Easy>')
            #duration_minutes: Duración en minutos
            beat_sequence: Lista de tuplas (init_beat_shift, note_type, pitch, duration_minutes, duration_beats, duration_beatshift)
                        note_type puede ser 'normal' o 'special'
            #{'position': 1536, 'type': 1, 'button': 2, 'duration': 0}
        Returns:
            Lista de IDs de tokens
        """
        tokens = [self.bos_token]
        
        # Agregar metadatos iniciales: instrumento y dificultad
        tokens.extend([instrument, difficulty])
        # Agregar secuencia de beats con la nueva estructura
        for init_beat_shift, note_type, pitch, duration_minutes, duration_beats, duration_beatshift in beat_sequence:
            if init_beat_shift <= self.max_beats and init_beat_shift <= self.max_beatshifts:
                # 1. Beatshift / note start
                tokens.append(f'<Beatshift_{init_beat_shift}>')

                # 2. Tipo de nota (normal o special)
                if note_type == 'normal':
                    tokens.append('<Normal>')
                elif note_type == 'special':
                    tokens.append('<Special>')
                
                # 3. Pitch según el instrumento
                if instrument == '<Drums>':
                    if 0 <= pitch <= 4:
                        tokens.append(f'<DrumsPitch_{pitch}>')
                else:  # Guitar o Bass
                    if 0 <= pitch <= 7:
                        tokens.append(f'<Pitch_{pitch}>')
                
                # 4. Duración en minutos
                tokens.append(f'<Minute_{duration_minutes}>')
                
                # 5. Duracion en Beat
                tokens.append(f'<Beat_{duration_beats}>')
                
                # 6. Beatshift Duracion Final
                tokens.append(f'<Beatshift_{duration_beatshift}>')
        
        tokens.append(self.eos_token)
        #return ' '.join(tokens)
        return self.convert_tokens_to_ids(tokens)

    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._convert_id_to_token(i) for i in ids]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            special_tokens = {self.bos_token, self.eos_token, self.pad_token}
            tokens = [t for t in tokens if t not in special_tokens]
        return ' '.join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
