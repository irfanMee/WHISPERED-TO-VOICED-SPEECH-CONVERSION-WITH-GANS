"""
This code reads a reads the noisy audio sample stored in folder named 'Noisy_testset' and runs through the SEGAN model and stores
the clean audio sample in folder named 'Clear_Sample'. 
"""


###Importing Libraries
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import soundfile as sf
from scipy.io import wavfile
from torch.autograd import Variable
import numpy as np
import random
import librosa
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import os


### Argument Parser
class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


### Main Function to run GAN
def GAN_Run(opts):

    #pre-check files
    assert opts.cfg_file is not None
    assert opts.test_files is not None
    assert opts.g_pretrained_ckpt is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))

    #CUDA check
    args.cuda = opts.cuda
    segan = SEGAN(args)  
    segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    if opts.cuda:
        segan.cuda()
    segan.G.eval()

    #Loading Input Files
    if opts.h5:
        with h5py.File(opts.test_files[0], 'r') as f:
            twavs = f['data'][:]
    else:
        file_location = os.path.join(TEST_FILES_PATH, '*.wav')
        print(file_location)
        twavs = glob.glob(file_location)
            

    #Cleaning Noise Audio
    print('Cleaning {} wavs'.format(len(twavs)))
    beg_t = timeit.default_timer()
    for t_i, twav in enumerate(twavs, start=1):
        if not opts.h5:
            tbname = os.path.basename(twav)
            rate, wav = wavfile.read(twav)
            wav = normalize_wave_minmax(wav)
        else:
            tbname = 'tfile_{}.wav'.format(t_i)
            wav = twav
            twav = tbname
        wav = pre_emphasize(wav, args.preemph)
        pwav = torch.FloatTensor(wav).view(1,1,-1)
        if opts.cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav)
        out_path = os.path.join(opts.synthesis_path,
                                tbname) 
        if opts.soundfile:
            sf.write(out_path, g_wav, 16000)
        else:
            wavfile.write(out_path, 16000, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t-beg_t))
        beg_t = timeit.default_timer()





### Main Function
if __name__ == "__main__":

    TEST_FILES_PATH='Noisy_testset/'
    MODEL_PATH="pretrained_model/train.opts"
    g_pretrained_ckpt = 'pretrained_model/segan_model.ckpt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default=g_pretrained_ckpt)
    parser.add_argument('--test_files', type=str, nargs='+', default=TEST_FILES_PATH)
    parser.add_argument('--h5', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--synthesis_path', type=str, default='Clear_Sample',
                        help='Path to save output samples (Def: ' \
                             'Clear_Sample).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--soundfile', action='store_true', default=False)
    parser.add_argument('--cfg_file', type=str, default=MODEL_PATH)

    opts, _  = parser.parse_known_args()

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)
    print(opts) 
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)
        
    GAN_Run(opts)