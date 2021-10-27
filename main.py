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
import hashlib
from pathvalidate import sanitize_filename
from django.core.validators import URLValidator
import youtube_dl
from contextlib import contextmanager, suppress
import warnings
import sys
import librosa
warnings.filterwarnings("ignore")
cpu = torch.device('cpu')
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
        if args.model != 'off':
            self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=channels)
            print('Loading checkpoint...',end=' ')
            self.demucs.to(device)
            self.demucs.load_state_dict(torch.load(demucs_name))
            print('done')
            self.demucs.eval()
        self.onnx_models = {}
        c = 0
        if args.onnx != 'off':
            self.models = get_models('tdf_extra', load=False, device=cpu, stems=args.stems)
            print(f'Loading onnx model{"s" if len(self.models) > 1 else ""}...',end=' ')
            for model in self.models:
                c+=1
                self.onnx_models[c] = ort.InferenceSession(os.path.join(args.onnx,model.target_name+'.onnx'))
            print('done')

        
        
    def prediction(self, m,b,d,o,v):
        file_paths = [b,d,o,v]    
        stems = ['bass',
                 'drums',
                 'others',
                 'vocals']
        #mix, rate = sf.read(m)
        mix, rate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix,mix])
        mix = mix.T
        sources = self.demix(mix.T)
        print('-'*20)
        print('Inferences finished!')
        print('-'*20)
        c = -1
        for i in sindex:
            c += 1
            print(f'Exporting {stems[i]}...',end=' ')
            if args.normalise:
                sources[i] = self.normalise(sources[i])
            sf.write(file_paths[i], sources[i].T, rate)
            print('done')
        if args.invert is not None:
            print('-'*20)
            for i in vindex:
                print('Inverting and exporting {}...'.format(stems[i]), end=' ')
                p = os.path.split(file_paths[i])
                sf.write(os.path.join(p[0],'invert_'+p[1]), (-sources[i].T)+mix, rate)
                print('done')
        print('-'*20)
    def normalise(self, wave):
        return wave / max(np.max(wave), abs(np.min(wave)))
    def dB_V(self, dB):
        return 10**(dB/20)
    def demix(self, mix):
        # 1 = demucs only
        # 0 = onnx only
        samples = mix.shape[-1]
        chunk_size = args.chunks*44100
        b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])
        segmented_mix = {}
        margin = args.margin
        if args.chunks == 0:
            chunk_size = int(mix.shape[-1])
        

        c = 0
        for skip in range(0, samples, chunk_size):
            c+=1
            end = min(skip+chunk_size+margin, samples)
            s_margin = 0 if c == 1 else margin
            e_margin = skip-s_margin
            segmented_mix[skip] = mix[:,e_margin:end].copy()
        segmented_mix = list(segmented_mix.values())

        if args.model == 'off' and args.onnx != 'off':
            sources = self.demix_base(segmented_mix, sindex, margin_size=margin)

        elif args.model != 'off' and args.onnx == 'off':
            sources = self.demix_demucs(segmented_mix, margin_size=margin)

        else: # both, apply spec effects in condition
            base_out = self.demix_base(segmented_mix, sindex, margin_size=margin)
            demucs_out = self.demix_demucs(segmented_mix, margin_size=margin)
            demucs_out, base_out = np.nan_to_num(demucs_out), np.nan_to_num(base_out)
            sources = {}
            for s in zip(sindex,range(len(b)-(len(sindex)-len(b)))):
                if not 'off' in [args.model,args.onnx]:
                    print(f'Using ratio: {b[s[0]]}')
                sources[s[0]] = (spec_effects(wave=[demucs_out[s[0]],base_out[s[1]]],
                                            algorithm=args.mixing,
                                            value=b[s[0]])*args.compensate) # compensation
        return sources
    def demix_base(self, mixes, sindex, margin_size):
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
            c += 1
            sources = []
            n_sample = mix.shape[1]
            mod = 0
            for model in self.models:
                mod += 1
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
                mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(cpu)
                with torch.no_grad():
                    _ort = self.onnx_models[mod]
                    #_ort = ort.InferenceSession(os.path.join(args.onnx,model.target_name+'.onnx'))
                    spek = model.stft(mix_waves)
                    
                    tar_waves = model.istft(torch.tensor(_ort.run(None, {'input': spek.cpu().numpy()})[0]))#.cpu()

                    tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).numpy()[:, :-pad]
                    s_margin = None if c == 1 else margin_size
                    e_margin = None if c == len(mixes) else -margin_size   
                    sources.append(tar_signal[:,s_margin:e_margin])
                progress_bar.update(1)
            
            chunked_sources[c] = sources
        _sources = concat_sources(chunked_sources)
        progress_bar.close()
        print(' >> done\n')
        del self.onnx_models
        return np.array(_sources)
    
    def demix_demucs(self, mix, margin_size):
        a = {}
        #counter = 0
        progress_bar = tqdm(total=len(mix))
        progress_bar.set_description("Processing demucs")
        chunk = 0
        for nmix in mix:
            chunk += 1
            nmix = torch.tensor(nmix, dtype=torch.float32)
            ref = nmix.mean(0)        
            nmix = (nmix - ref.mean()) / ref.std()
            
            with torch.no_grad():
                sources = apply_model(self.demucs, nmix.to(device), split=True, overlap=args.overlap, shifts=args.shifts)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            s_margin = None if chunk == 1 else margin_size
            e_margin = None if chunk == len(mix) else -margin_size
            a[chunk] = sources[:,:,s_margin:e_margin]
            progress_bar.update(1)
        sources = list(a.values())
        sources = np.concatenate(sources, axis=-1)
        progress_bar.close()
        print(' >> done\n')
        return sources

def downloader(link, supress=False, dl=False):
    validate = URLValidator()
    try:
        validate(link)
        inputsha = hashlib.sha1(bytes(link, encoding='utf8')).hexdigest() + '.wav'
        fmt = '251/140/250/139' if 'youtu' in link else 'best'
        s = 'YouTube link' if 'youtu' in link else 'Link'
        opt = {'format': fmt, 'outtmpl': inputsha, 'updatetime': False, 'nocheckcertificate': True}
        if not supress:
            print(f'{s} detected.\nAttempting to download...',end=' ')
        
        with hide_opt(), youtube_dl.YoutubeDL(opt) as ydl:
            ## dowload n take youtube info
            desc = ydl.extract_info(link, download=not os.path.isfile(inputsha) or dl)

        titlename = sanitize_filename(desc['title'])
        if not supress:
            print('done\n'+titlename)
        return [inputsha,titlename]
    except:
        return [link]

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
    p.add_argument('--margin',default=44100, type=int,
                              help='margin between chunks')

    p.add_argument('--invert','-inv', type=str, default=None,
                              help='invert stems to mixture. Ex: \'-inv v\' to get mixture-vocal difference.')

    #experimental
    p.add_argument('--compensate', type=float, default=1)

    p.add_argument('--channel','-c', type=int, default=64)
    p.add_argument('--overlap','-ov', type=float, default=0.5)
    args = p.parse_args()

    autoDL = downloader(args.input)
    isLink = False
    args.input = autoDL[0]
    if len(autoDL) == 2:
        isLink = True
    _basename = os.path.splitext(os.path.basename(args.input))[0]
    if not os.path.exists(os.path.join(args.output,_basename)):
        os.makedirs(os.path.join(args.output,_basename))
    if args.model == 'off' and args.onnx == 'off':
        print('Not so sure what model to use huh? 😉')
    if args.invert is not None and args.normalise:
        print('Inverting stems with normalise flag is not advised.')
    #some krazy A.I here 😎 dun judge my code plzzz lol
    global sindex, vindex
    sindex,vindex = [],[]
    if 'b' in args.stems:
        sindex.append(0)
        if 'b' in args.invert:
            vindex.append(0)
    if 'd' in args.stems:
        sindex.append(1)
        if 'd' in args.invert:
            vindex.append(1)
    if 'o' in args.stems:
        sindex.append(2)
        if 'o' in args.invert:
            vindex.append(2)
    if 'v' in args.stems:
        sindex.append(3)
        if 'v' in args.invert:
            vindex.append(3)
    stems = [
        'bass.wav',
        'drums.wav',
        'other.wav',
        'vocals.wav'
    ]
    if args.onnx != 'off':
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

    if isLink:
        os.rename(os.path.join(args.output,_basename),
                  os.path.join(args.output,autoDL[1]))
        if os.path.isfile(args.input):
            os.remove(args.input)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Successfully completed music demixing.");print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
