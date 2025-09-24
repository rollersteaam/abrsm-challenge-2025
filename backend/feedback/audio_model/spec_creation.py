import torchaudio
import os 
import numpy as np
import openl3
import torchaudio
import soundfile as sf

def create_melspec(waveform, sr=22050):
    """
    Create mel spectrogram using the specified parameters
    Always resample to 22kHz for consistent frame rates
    """
    # Resample to 22kHz if needed
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resampler(waveform)
    
    melspect_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, 
        n_fft=1024, 
        hop_length=441, 
        f_min=30, 
        f_max=11000, 
        n_mels=128, 
        mel_scale='slaney', 
        normalized='frame_length', 
        power=1
    )
    melspect = melspect_transform(waveform).mul(1000).log1p()  # Add small value to avoid log(0)
    
    return melspect

def create_song_embedding(file_path):
    audio, sr = sf.read(file_path)  # waveform and sampling rate
    # Get embedding
    emb, ts = openl3.get_audio_embedding(audio, sr, hop_size=1.0)
    print(emb.shape)
    return emb

def process_audio_file(file_path):
    waveform, sr = torchaudio.load(file_path)
    melspec = create_melspec(waveform, sr)
    return melspec


songs = sorted(os.listdir('/Users/acw707/Documents/abrsm_lmth25/audio/'))
num_songs = len(songs)
spec_dict = {}
idx = 0
for song in songs:
    if not song.endswith('.mp3'):
        continue
    print(f"Processing {idx+1}/{num_songs}")
    print(f"Song: {song}")
    song_name  = song[:-4]
    file_path = os.path.join('/Users/acw707/Documents/abrsm_lmth25/audio/', song)
    #melspec = process_audio_file(file_path)
    melspec = create_song_embedding(file_path)
    spec_dict[song_name] = melspec
    idx += 1

np.savez('/Users/acw707/Documents/abrsm_lmth25/data/emb_dict.npz', **spec_dict)