import glob
import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import spectral_normalize_torch, MAX_WAV_VALUE
from models import Generator
from tqdm import tqdm

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--mel-dir', type=str, required=True)
    parser.add_argument('--checkpoint-file', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, default='generated_audios')

    args = parser.parse_args()
    return args

def inference(args, generator, mel_dir: str):
    filelist = sorted(os.listdir(mel_dir))

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for idx, fname in tqdm(enumerate(filelist)):
            mel_path = os.path.join(mel_dir, fname)
            mel = np.load(mel_path)
            mel = torch.from_numpy(mel).unsqueeze(0)
            mel = spectral_normalize_torch(mel)
            mel = mel.to(device)

            y_g_hat = generator(mel)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(args.output_dir, os.path.splitext(fname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
    

def main():
    args = parse_args()

    config_file = os.path.join(os.path.split(args.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    inference(args, generator, args.mel_dir)


if __name__ == '__main__':
    main()