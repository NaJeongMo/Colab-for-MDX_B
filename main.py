# this module seems messy, you can't import this module and use it with another script
# It's using a global variable (args) for argument parsing.
from numpy.lib import ediff1d, source
import soundfile as sf
import torch
import numpy as np
from demucs.model import Demucs
from demucs.utils import apply_model
from models import get_models, spec_effects
import onnxruntime as ort
import time
import argparse
import os
from tqdm import tqdm
from contextlib import contextmanager
import warnings
import sys
import librosa
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@contextmanager
def hide_opt():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class Predictor:
    def __init__(self):
        pass
    def prediction_setup(self, demucs_name,
                               channels=64):
        self.models = get_models('tdf_extra', load=False, device=device, stems=args.stems)
        if args.model != 'off':
            self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=channels)
            print('Loading checkpoint...',end=' ')
            self.demucs.to(device)
            self.demucs.load_state_dict(torch.load(demucs_name))
            print('done')
            self.demucs.eval()
        
        
    def prediction(self, m,b,d,o,v):
        file_paths = [b,d,o,v]    
        stems = ['bass',
                 'drums',
                 'others',
                 'vocals']
        #mix, rate = sf.read(m)
        mix, rate = librosa.load(m, mono=False, sr=44100)
        mix = mix.T
        sources = self.demix(mix.T)
        print('-'*30)
        print('Inferences finished!')
        print('-'*30)
        c = -1
        for i in zip(range(len(sources)), sindex):
            c += 1
            print(f'Exporting {stems[i[1]]}...',end=' ')
            if args.normalise:
                sources[i[0]] = self.normalise(sources[i[0]])
            sf.write(file_paths[i[1]], sources[i[0]].T, rate)
            print('done')
    def normalise(self, wave):
        return wave / max(np.max(wave), abs(np.min(wave)))
    def demix(self, mix):
        # 1 = demucs only
        # 0 = onnx only
        samples = mix.shape[-1]
        chunk_size = args.chunks*44100
        b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])
        segmented_mix = {}
        if args.chunks == 0:
            chunk_size = int(mix.shape[-1])
        for skip in range(0, samples, chunk_size):
            end = min(skip+(chunk_size), samples)
            segmented_mix[skip] = mix[:,skip:end].copy()
        segmented_mix = list(segmented_mix.values())

        if args.model == 'off' and args.onnx != 'off':
            sources = self.demix_base(segmented_mix, sindex)
        elif args.model != 'off' and args.onnx == 'off':
            sources = self.demix_demucs(segmented_mix)
        else: # both, apply spec effects in condition
            demucs_out = self.demix_demucs(segmented_mix)
            base_out = self.demix_base(segmented_mix, sindex)
            sources = []
            for s in zip(sindex,range(len(b)-(len(sindex)-len(b)))):
                print(f'Using ratio: {b[s[0]]}')
                sources.append(spec_effects(wave=[demucs_out[s[0]],base_out[s[1]]],
                                            algorithm=args.mixing,
                                            value=b[s[0]]))
        return sources
    def demix_base(self, mixes, sindex):
        def concat_sources(chunked_sources):
            sources = []
            for s in range(len(sindex)):
                source = []
                for chunk in range(1, len(chunked_sources)+1):
                    source.append(chunked_sources[chunk][s])
                sources.append(np.concatenate(source, axis=-1))
            return sources
        chunked_sources = {}
        c = 0
        progress_bar = tqdm(total=len(mixes)*len(self.models))
        progress_bar.set_description("Processing base")
        for mix in mixes:
            sources = []
            n_sample = mix.shape[1]
            for model in self.models:
                trim = model.n_fft//2
                gen_size = model.chunk_size-2*trim
                pad = gen_size - n_sample%gen_size
                mix_p = np.concatenate((np.zeros((2,trim)), mix, np.zeros((2,pad)), np.zeros((2,trim))), 1)
                mix_waves = []
                i = 0
                while i < n_sample + pad:
                    waves = np.array(mix_p[:, i:i+model.chunk_size])
                    mix_waves.append(waves)
                    i += gen_size
                mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(device)
                with torch.no_grad():
                    _ort = ort.InferenceSession(os.path.join(args.onnx,model.target_name+'.onnx'))
                    spek = model.stft(mix_waves)
                    
                    tar_waves = model.istft(torch.tensor(_ort.run(None, {'input': spek.cpu().numpy()})[0]).to(device)).cpu()

                    tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).numpy()[:, :-pad]
                
                    sources.append(tar_signal)
                progress_bar.update(1)
            c += 1
            chunked_sources[c] = sources
        _sources = concat_sources(chunked_sources)
        progress_bar.close()
        print(' >> done\n')
        return np.array(_sources)
    
    def demix_demucs(self, mix):
        a = {}
        #counter = 0
        progress_bar = tqdm(total=len(mix))
        progress_bar.set_description("Processing demucs")
        for nmix in mix:
            nmix = torch.tensor(nmix, dtype=torch.float32)
            ref = nmix.mean(0)        
            nmix = (nmix - ref.mean()) / ref.std()
            
            with torch.no_grad():
                sources = apply_model(self.demucs, nmix.to(device), split=True, overlap=args.overlap, shifts=args.shifts)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]
            a[nmix] = sources
            progress_bar.update(1)
        sources = list(a.values())
        sources = np.concatenate(sources, axis=-1)
        progress_bar.close()
        print(' >>done\n')
        return sources
def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default='model/demucs_extra.ckpt',
                              help='Demucs checkpoint path')
    p.add_argument('--onnx','-O', default='onnx',
                              help='ONNX Model path')
    p.add_argument('--input', '-i', type=str, required=True)
    p.add_argument('--output','-o', default='separated/',
                              help='Output path')
    p.add_argument('--shifts','-S', default=0, type=int,
                              help='Predict with randomised equivariant stabilisation')
    p.add_argument('--mixing','-M', default='default', choices=['default','min_mag','max_mag'],
                              help='Mixing type')
    p.add_argument('--normalise','-n', default=False, action='store_true',
                              help='Normalise stems')
    p.add_argument('--stems', '-s', default='bdov')
    p.add_argument('--chunks','-C', default=1, type=int,
                              help='Split input files into chunks for lower ram utilisation')

    #experimental
    p.add_argument('--channel','-c', type=int, default=64)
    p.add_argument('--overlap','-ov', type=float, default=0.25)
    args = p.parse_args()

    _basename = os.path.splitext(os.path.basename(args.input))[0]

    if not os.path.exists(os.path.join(args.output,_basename)):
        os.makedirs(os.path.join(args.output,_basename))
    global sindex
    sindex = []
    if 'b' in args.stems:
        sindex.append(0)
    if 'd' in args.stems:
        sindex.append(1)
    if 'o' in args.stems:
        sindex.append(2)
    if 'v' in args.stems:
        sindex.append(3)
    stems = [
        'bass.wav',
        'drums.wav',
        'other.wav',
        'vocals.wav'
    ]
    for c in sindex:
        if not os.path.isfile(os.path.join(args.onnx,os.path.splitext(stems[c])[0])+'.onnx'):
            raise FileNotFoundError(f'{os.path.splitext(stems[c])[0]}.onnx not found')
    output = lambda x, stem: os.path.join(x,stems[stem])

    e = os.path.join(args.output,_basename)

    pred = Predictor()
    pred.prediction_setup(demucs_name=args.model,
                          channels=args.channel)
    
    # split
    pred.prediction(
        m=args.input,
        
        b=output(e, 0),
        d=output(e, 1),
        o=output(e, 2),
        v=output(e, 3)

    )


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Successfully completed music demixing.");print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
