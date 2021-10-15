# this module seems messy, you can't import this module and use it with another script
# It's using a global variable (args) for argument parsing.
from numpy.lib import ediff1d
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
from contextlib import contextmanager
import warnings
import sys
warnings.filterwarnings("ignore")

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
        self.models = get_models('tdf_extra', load=False, device=torch.device('cpu'))

        if args.model != 'off':
            self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=channels)
            print('Loading checkpoint...',end=' ')
            self.demucs.load_state_dict(torch.load(demucs_name))
            print('done')
            self.demucs.eval()
        
        
    def prediction(self, m,b,d,o,v):
        file_paths = [b,d,o,v]    
        stems = ['bass',
                 'drums',
                 'others',
                 'vocals']
                 
        mix, rate = sf.read(m)
        sources = self.demix(mix.T)
        print('-'*30)
        print('Inferences finished!')
        print('-'*30)
        for i in range(len(sources)):
            print(f'Exporting {stems[i]}...',end=' ')
            sf.write(file_paths[i], sources[i].T, rate)
            print('done')
    def normalise(self, wave):
        return wave / max(np.max(wave), abs(np.min(wave)))
    def demix(self, mix):
        # 1 = demucs only
        # 0 = onnx only
        b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.7]]])
        if args.model == 'off' and args.onnx != 'off':
            sources = self.demix_base(mix)
        elif args.model != 'off' and args.onnx == 'off':
            sources = self.demix_demucs(mix)
        else: # both, apply spec effects in condition
            demucs_out = self.demix_demucs(mix)
            base_out = self.demix_base(mix)
            sources = []
            for s in zip(range(len(b)),b):
                sources.append(spec_effects(wave=[demucs_out[s[0]],base_out[s[0]]],
                                            algorithm=args.mixer,
                                            value=s[1]))
        if args.normalise:
            for s in range(len(b)):
                sources[s] = self.normalise(sources[s])
        return sources
    def demix_base(self, mix):
        sources = []
        n_sample = mix.shape[1]
        for model in self.models:
            print(f'Inference session {model.target_name}...')
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
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32)
            #print(mix_waves.shape)
            with torch.no_grad():
                _ort = ort.InferenceSession(os.path.join(args.onnx,model.target_name+'.onnx'))
                spec = model.stft(mix_waves).numpy()
                
                tar_waves = model.istft(torch.tensor(
                    _ort.run(None, {'input': spec})[0]
                ))
                tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).numpy()[:, :-pad]
            
                sources.append(tar_signal)
        return np.array(sources)
    
    def demix_demucs(self, mix):
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        
        with torch.no_grad():
            print('Applying demucs model...', end=' ')
            sources = apply_model(self.demucs, mix, split=True, overlap=args.overlap, shifts=args.shifts)
            print('done\n')
            
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
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
    p.add_argument('--shifts','-S', default=0,
                              help='Predict with randomised equivariant stabilisation')
    p.add_argument('--mixer','-M', default='default', choices=['default','min_mag','max_mag'],
                              help='Mixing type')
    p.add_argument('--normalise','-n', default=False, action='store_true',
                              help='Normalise stems')
    #p.add_argument('--chunks','-C', default=1,
    #                          help='Split input files into chunks for lower ram utilisation')

    #experimental
    p.add_argument('--channel','-c', default=64)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--overlap','-ov', default=0.25)
    args = p.parse_args()

    _basename = os.path.splitext(os.path.basename(args.input))[0]

    if not os.path.exists(os.path.join(args.output,_basename)):
        os.makedirs(os.path.join(args.output,_basename))

    stems = [
        'bass.wav',
        'drums.wav',
        'other.wav',
        'vocals.wav'
    ]

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
