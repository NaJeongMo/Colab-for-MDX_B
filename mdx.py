import torch
import onnxruntime as ort
from tqdm import tqdm
import warnings
import numpy as np
import hashlib

warnings.filterwarnings("ignore")

class MDX_Model:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft//2+1
        self.chunk_size = hop * (self.dim_t-1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)

        out_c = self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins-self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,4,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        # c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,2,self.chunk_size])


class MDX:

    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path:str, params:MDX_Model, processor=DEFAULT_PROCESSOR):

        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']

        self.model = params
        # self.model = MDX_Model(
        #     self.device,
        #     dim_f = 2048,
        #     dim_t = 256,
        #     n_fft = 2048*3 # dim_f * n_fft_scale[target_name]
        # )
        # )
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        self.process = lambda spec:self.ort.run(None, {'input': spec.cpu().numpy()})[0]

    @staticmethod
    def get_hash(model_path):

        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
            
        return model_hash
    
    # Segment or join segmented wave array
    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE, sr=DEFAULT_SR) -> list:
        
        if combine:
            processed_wave = None  # Initializing as None instead of [] for later numpy array concatenation
            for segment_count, segment in enumerate(wave):
                start = 0 if not segment_count else margin_size
                end = None if segment_count == len(wave) else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):

                margin = 0 if not segment_count else margin_size
                end = min(skip+chunk_size+margin, sample_count)
                start = skip-margin

                cut = wave[:,start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break
        
        return processed_wave
    
    def process_wave(self, wave:np.array):
        
        def pad_wave(wave):
            n_sample = wave.shape[1]
            trim = self.model.n_fft//2
            gen_size = self.model.chunk_size-2*trim
            pad = gen_size - n_sample%gen_size

            # padded wave
            wave_p = np.concatenate((np.zeros((2,trim)), wave, np.zeros((2,pad)), np.zeros((2,trim))), 1)

            mix_waves = []
            for i in range(0, n_sample+pad, gen_size):
                waves = np.array(wave_p[:, i:i+self.model.chunk_size])
                mix_waves.append(waves)

            mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

            return mix_waves, pad, trim

        mix_waves, pad, trim = pad_wave(wave)
        mix_wavesx = mix_waves.split(1)

        with torch.no_grad():
            pw = []
            for mix_waves in tqdm(mix_wavesx):

                spec = self.model.stft(mix_waves)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:,:,trim:-trim].transpose(0,1).reshape(2, -1).cpu().numpy()#[:, :-pad]
                pw.append(processed_wav)
        
        return np.concatenate(pw, axis=-1)
