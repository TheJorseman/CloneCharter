from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
import torchaudio
import torchaudio.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MERT:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type=='cuda' else torch.float32
        ).to(device)
        self.model.eval()

        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True
        )

        self._resampler = None
        self._orig_sr = None

        self.aggregator = torch.nn.Conv1d(
            in_channels=13, out_channels=1, kernel_size=1
        ).to(device)
        if device.type == 'cuda':
            self.aggregator = self.aggregator.half()

    def _maybe_resample(self, waveform, sr):
        target_sr = self.fe.sampling_rate
        if sr != target_sr:
            if self._resampler is None or self._orig_sr != sr:
                self._resampler = T.Resample(sr, target_sr)
                self._orig_sr = sr
            waveform = self._resampler(waveform)
        return waveform

    def forward(self, audio_file):
        waveform, sr = torchaudio.load(audio_file)
        waveform = self._maybe_resample(waveform, sr)
        
        inp = self.fe(
            waveform.flatten(),
            sampling_rate=self.fe.sampling_rate,
            return_tensors='pt'
        )
        
        for k, v in inp.items():
            if v.is_floating_point() and device.type == 'cuda':
                inp[k] = v.to(device).half()
            else:
                inp[k] = v.to(device)

        with torch.no_grad():
            outputs = self.model(**inp, output_hidden_states=True)
            hidden = outputs.hidden_states

        # FIX: Manejar correctamente las dimensiones
        stacked = torch.stack(hidden)  # [13, batch, seq_len, 768]
        
        # Squeeze para eliminar dimensiones extra si batch_size=1
        stacked = stacked.squeeze()
        
        # Verificar dimensiones y ajustar si es necesario
        if stacked.dim() == 4:  # [13, 1, seq_len, 768]
            stacked = stacked.squeeze(1)  # [13, seq_len, 768]
        elif stacked.dim() == 3 and stacked.size(0) != 13:
            # Si la primera dimensión no es 13, reorganizar
            stacked = stacked.permute(1, 0, 2)  # Reordenar si es necesario
            
        # Reducir dimensión temporal (promedio sobre seq_len)
        time_reduced = stacked.mean(-2)  # [13, 768]
        
        # Asegurar que tenemos las dimensiones correctas para Conv1d
        if time_reduced.dim() == 1:
            time_reduced = time_reduced.unsqueeze(0)  # [1, 768] -> [1, 1, 768]
            time_reduced = time_reduced.unsqueeze(0)  # [1, 1, 768]
        elif time_reduced.dim() == 2:
            time_reduced = time_reduced.unsqueeze(0)  # [13, 768] -> [1, 13, 768]
            
        # Aplicar agregación
        x = self.aggregator(time_reduced).squeeze()  # [768]
        
        return x.float().cpu().tolist()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def from_tensor(self, waveform, sr):
        if waveform.ndim > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = self._maybe_resample(waveform, sr)
        
        inp = self.fe(
            waveform.flatten(), 
            sampling_rate=self.fe.sampling_rate, 
            return_tensors='pt'
        )
        
        for k, v in inp.items():
            if v.is_floating_point() and device.type == 'cuda':
                inp[k] = v.to(device).half()
            else:
                inp[k] = v.to(device)

        with torch.no_grad():
            outputs = self.model(**inp, output_hidden_states=True)
            hidden = outputs.hidden_states

        # Mismo manejo de dimensiones que en forward()
        stacked = torch.stack(hidden).squeeze()
        
        if stacked.dim() == 4:
            stacked = stacked.squeeze(1)
        elif stacked.dim() == 3 and stacked.size(0) != 13:
            stacked = stacked.permute(1, 0, 2)
            
        time_reduced = stacked.mean(-2)
        
        if time_reduced.dim() == 1:
            time_reduced = time_reduced.unsqueeze(0).unsqueeze(0)
        elif time_reduced.dim() == 2:
            time_reduced = time_reduced.unsqueeze(0)
            
        x = self.aggregator(time_reduced).squeeze()
        return x.float().cpu().tolist()
