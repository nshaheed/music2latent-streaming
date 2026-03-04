import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import librosa
import matplotlib.pyplot as plt
from dasp_pytorch.functional import compressor

from scipy.signal import hilbert, butter, filtfilt
from .hparams import hparams


def wv2spec(wv, hop_size=256, fac=4):
    X = stft(wv, hop_size=hop_size, fac=fac, device=wv.device)
    X = power2db(torch.abs(X)**2)
    X = normalize(X)
    return X

def spec2wv(S,P, hop_size=256, fac=4):
    S = denormalize(S)
    S = torch.sqrt(db2power(S))
    P = P * np.pi
    SP = torch.complex(S * torch.cos(P), S * torch.sin(P))
    return istft(SP, fac=fac, hop_size=hop_size, device=SP.device)

def denormalize_realimag(x):
    x = x/hparams.beta_rescale
    return torch.sign(x)*(x.abs()**(1./hparams.alpha_rescale))

def normalize_complex(x):
    return hparams.beta_rescale*(x.abs()**hparams.alpha_rescale).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

def denormalize_complex(x):
    x = x/hparams.beta_rescale
    return (x.abs()**(1./hparams.alpha_rescale)).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

def wv2complex(wv, hop_size=256, fac=4):
    X = stft(wv, hop_size=hop_size, fac=fac, device=wv.device)
    return X[:,:hop_size*2,:]

def wv2realimag(wv, hop_size=256, fac=4):
    X = wv2complex(wv, hop_size, fac)
    X = normalize_complex(X)
    return torch.stack((torch.real(X),torch.imag(X)), -3)

def realimag2wv(x, hop_size=256, fac=4):
    x = torch.nn.functional.pad(x, (0,0,0,1))
    real,imag = torch.chunk(x, 2, -3)
    X = torch.complex(real.squeeze(-3),imag.squeeze(-3))
    X = denormalize_complex(X)
    return istft(X, fac=fac, hop_size=hop_size, device=X.device).clamp(-1.,1.)

def to_representation_encoder(x):
    return wv2realimag(x, hparams.hop)

def to_representation(x):
    return wv2realimag(x, hparams.hop)

def to_waveform(x):
    return realimag2wv(x, hparams.hop)

def overlap_and_add(signal, frame_step):

    outer_dimensions = signal.shape[:-2]
    outer_rank = torch.numel(torch.tensor(outer_dimensions))

    def full_shape(inner_shape):
      s = torch.cat([torch.tensor(outer_dimensions), torch.tensor(inner_shape)], 0)
      s = list(s)
      s = [int(el) for el in s]
      return s

    frame_length = signal.shape[-1]
    frames = signal.shape[-2]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    signal = torch.nn.functional.pad(signal, (0, segments * frame_step - frame_length, 0, segments))

    shape = full_shape([frames + segments, segments, frame_step])
    signal = torch.reshape(signal, shape)

    perm = torch.cat([torch.arange(0, outer_rank), torch.tensor([el+outer_rank for el in [1, 0, 2]])], 0)
    perm = list(perm)
    perm = [int(el) for el in perm]
    signal = torch.permute(signal, perm)

    shape = full_shape([(frames + segments) * segments, frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal[..., :(frames + segments - 1) * segments, :]

    shape = full_shape([segments, (frames + segments - 1), frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal.sum(-3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step])
    signal = torch.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_length]

    return signal

def inverse_stft_window(frame_length, frame_step, forward_window):
    denom = forward_window**2
    overlaps = -(-frame_length // frame_step)
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = torch.reshape(denom, [overlaps, frame_step])
    denom = torch.sum(denom, 0, keepdim=True)
    denom = torch.tile(denom, [overlaps, 1])
    denom = torch.reshape(denom, [overlaps * frame_step])
    return forward_window / denom[:frame_length]

def istft(SP, fac=4, hop_size=256, device='cuda'):
    x = torch.fft.irfft(SP, dim=-2)
    window = torch.hann_window(fac*hop_size).to(device)
    window = inverse_stft_window(fac*hop_size, hop_size, window)
    x = x*window.unsqueeze(-1)
    return overlap_and_add(x.permute(0,2,1), hop_size)

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames

def stft(wv, fac=4, hop_size=256, device='cuda'):
    window = torch.hann_window(fac*hop_size).to(device)
    framed_signals = frame(wv, fac*hop_size, hop_size)
    framed_signals = framed_signals*window
    return torch.fft.rfft(framed_signals, n=None, dim=- 1, norm=None).permute(0,2,1)

def normalize(S, mu_rescale=-25., sigma_rescale=75.):
    return (S - mu_rescale) / sigma_rescale

def denormalize(S, mu_rescale=-25., sigma_rescale=75.):
    return (S * sigma_rescale) + mu_rescale

def db2power(S_db, ref=1.0):
    return ref * torch.pow(10.0, 0.1 * S_db)

def power2db(power, ref_value=1.0, amin=1e-10):
    log_spec = 10.0 * torch.log10(torch.maximum(torch.tensor(amin), power))
    log_spec -= 10.0 * torch.log10(torch.maximum(torch.tensor(amin), torch.tensor(ref_value)))
    return log_spec

def create_melmat(hop=256, mel_bins=256, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    melmat_pt = torchaudio.functional.melscale_fbanks(int((4*hop) // 2 + 1), n_mels=mel_bins, f_min=0.0, f_max=hparams.sample_rate / 2.0, sample_rate=hparams.sample_rate)
    mel_f = torch.from_numpy(librosa.mel_frequencies(n_mels=mel_bins + 2, fmin=0., fmax=hparams.sample_rate//2))
    enorm = (2.0 / (mel_f[2 : mel_bins + 2] - mel_f[:mel_bins])).unsqueeze(0).to(torch.float32)
    melmat_pt = torch.mul(melmat_pt, enorm)
    melmat_pt = torch.div(melmat_pt, torch.sum(melmat_pt, dim=0))
    melmat_pt[torch.isnan(melmat_pt)] = 0
    return melmat_pt.to(device)

def wv2mel(x):
    melmat = create_melmat()
    melmat = melmat.to(x.device)
    return torch.tensordot(wv2spec(x), melmat, dims=([-2],[0])).permute(0,2,1)

def plot_audio(wv):

    # wv has shape [batch_size, samples]

    spec = wv2mel(wv)
    fig, axs = plt.subplots(nrows=spec.shape[0], ncols=1)
    for ind in range(spec.shape[0]):
        axs[ind].imshow(np.flip(spec.cpu().numpy(), -2), cmap=None)
        axs[ind].axis('off')
        axs[ind].set_title('Mel-Spectrogram')
    return fig

# def plot_audio_compare(wv1,wv2):
#     spec1 = []
#     spec2 = []
#     for w1,w2 in zip(wv1,wv2):
#         spec1.append(wv2mel(w1.unsqueeze(0)).squeeze(0)[..., :1024])
#         spec2.append(wv2mel(w2.unsqueeze(0)).squeeze(0)[..., :1024])

#     fig, axs = plt.subplots(nrows=len(spec1), ncols=2, figsize=(5*len(spec1),10))

#     for ind in range(len(spec1)):

#         axs[ind][0].imshow(np.flip(spec1[ind].cpu().numpy(), -2), cmap=None)
#         axs[ind][0].axis('off')

#         axs[ind][1].imshow(np.flip(spec2[ind].cpu().numpy(), -2), cmap=None)
#         axs[ind][1].axis('off')
#     return fig

def plot_training_run(wv, flat_wv, env_wv):
    spec_wv = wv2mel(wv)
    spec_flat_wv = wv2mel(flat_wv)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))

    # breakpoint()
    axs[0].imshow(np.flip(spec_wv[0].cpu().numpy(), -2))
    axs[0].axis('off')
    axs[1].imshow(np.flip(spec_flat_wv[0].cpu().numpy(), -2))
    axs[1].axis('off')

    # 1. Define the x-axis based on the original waveform length
    x_orig = np.arange(wv.shape[-1])

    # 2. Define the x-axis for the current envelope (spread across the same range)
    # We map the indices of env_wv to the full range of x_orig
    x_env_current = np.linspace(0, wv.shape[-1] - 1, num=env_wv.shape[-1])

    # 3. Interpolate env_wv to match the length of wv
    env_interp = np.interp(x_orig, x_env_current, env_wv[0][0].cpu().numpy())

    # axs[2].plot(np.arange(wv.shape[-1]), wv[0].cpu().numpy())
    # axs[3].plot(np.arange(env_wv.shape[-1]), env_wv[0][0].cpu().numpy())

    axs[2].plot(x_orig, wv[0].cpu().numpy(), label='Waveform', alpha=0.7)
    axs[2].plot(x_orig, env_interp, label='Envelope', linewidth=2)
    axs[2].legend()

    return fig

def plot_audio_compare(wv1, wv2):
    spec1 = []
    spec2 = []
    spec3 = []
    envs = []
    wv3 = []

    for w1, w2 in zip(wv1, wv2):
        spec1.append(wv2mel(w1.unsqueeze(0)).squeeze(0)[..., :1024])
        spec2.append(wv2mel(w2.unsqueeze(0)).squeeze(0)[..., :1024])

        # breakpoint()
        _, flattened_w1 = extract_spectrum(w1.unsqueeze(0))
        spec3.append(wv2mel(flattened_w1).squeeze(0)[..., :1024])
        wv3.append(flattened_w1)

        aud_env, smooth_env, env = extract_envelope(w1.unsqueeze(0), target_length=1024, cutoff_freq=30)
        envs.append(env)
        # breakpoint()

    fig, axs = plt.subplots(nrows=len(spec1), ncols=5, figsize=(5 * len(spec1), 10))

    for ind in range(len(spec1)):
        axs[ind][0].imshow(np.flip(spec1[ind].cpu().numpy(), -2))
        axs[ind][0].axis('off')


        axs[ind][1].imshow(np.flip(spec2[ind].cpu().numpy(), -2))
        axs[ind][1].axis('off')

        axs[ind][3].imshow(np.flip(spec3[ind].cpu().numpy(), -2))
        axs[ind][3].axis('off')

        # envelope column
        # axs[ind][2].imshow(envs[ind].cpu().numpy()[None, :])
        # breakpoint()
        axs[ind][2].plot(np.arange(envs[ind].shape[-1]), envs[ind][-1].cpu().numpy())
        # axs[ind][2].axis('off')

        axs[ind][4].plot(np.arange(wv3[ind].shape[-1]), wv3[ind][-1].cpu().numpy())

    return fig

def extract_envelope(wv, target_length=64, cutoff_freq=30):
    """
    Extract a fixed-length temporal envelope from an audio file.

    Parameters:
    -----------
    filepath : str
        Path to the WAV file

    target_length : int
        Desired length of the output envelope (default: 2048)

    cutoff_freq : float
        Cutoff frequency for the lowpass filter in Hz (default: 30)


    Returns:
    --------
    tuple
        (original_envelope, smoothed_envelope, resampled_envelope)
    """
    # sample_rate, audio = wavfile.read(filepath)
    device = wv.device
    audio = wv.cpu().numpy()

    # Convert to mono if stereo (shouldn't need)
    # if len(audio.shape) > 1:
    #     audio = np.mean(audio, axis=1)

    # Normalize audio
    # audio = audio.astype(float) / np.max(np.abs(audio))

    # Calculate temporal envelope using Hilbert transform
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)

    # assume 44.1kHz sample rate (this won't matter too much in reality as:
    #  normalized_cutoff = 30 / 22050 = 0.00136
    #  normalized_cutoff = 30 / 24000 = 0.00125
    # this is only an 8.8% difference in cutoff frequency from 44.1k vs 48k
    sample_rate = 44100

    # Design and apply lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normalized_cutoff, btype='low')
    smoothed_envelope = filtfilt(b, a, envelope)

    # the nyquist frequency for 44100 is (86.13/2) = 43
    #                       for 48000 is (93.75/2) = 47
    # so I'm not too worried about aliasing


    # # Resample to target length
    # batch, samps = smoothed_envelope.shape

    # original_indices = np.linspace(0, samps - 1, samps)
    # target_indices = np.linspace(0, samps - 1, target_length)

    # resampled_envelope = np.interp(
    #     target_indices,
    #     original_indices,
    #     smoothed_envelope.reshape(-1, samps)
    # ).reshape(batch, target_length)

    # original_indices = np.linspace(0, len(smoothed_envelope)-1, len(smoothed_envelope))
    # target_indices = np.linspace(0, len(smoothed_envelope)-1, target_length)
    # resampled_envelope = np.interp(target_indices, original_indices, smoothed_envelope)

    batch, samps = smoothed_envelope.shape

    original_indices = np.linspace(0, samps - 1, samps)
    target_indices = np.linspace(0, samps - 1, target_length)

    resampled_envelope = np.empty((batch, target_length), dtype=smoothed_envelope.dtype)

    for b in range(batch):
        resampled_envelope[b] = np.interp(
            target_indices,
            original_indices,
            smoothed_envelope[b]
        )

    # we only care about the resampled one for now
    resampled_envelope = torch.from_numpy(resampled_envelope).to(torch.float32).to(device)
    return envelope, smoothed_envelope, resampled_envelope


def extract_spectrum(audio):
    """
    Extract the spectrum of the audio by flattening it's envelope

    Parameters:
    -----------
    filepath : str
        Path to the WAV file
    target_length : int
        Desired length of the output envelope (default: 2048)
    cutoff_freq : float
        Cutoff frequency for the lowpass filter in Hz (default: 30)
    plot : bool
        Whether to create visualization plots (default: True)

    Returns:
    --------
    tuple
        (original_envelope, smoothed_envelope, resampled_envelope)
    """

    # question: who to flatten? Julius defines flattening the spectrum as:
    # dividing the spectrum of each carrier frame by its own spectral envelope, thereby flattening it.
    # however, this is applying the *spectral* envelope of the signal, but the paper I'm basing things
    # off of applies *temporal* envelopes. I think that this is good justification for trying a neural
    # net based approach rather than just fucking around like I'm doing now, but I'm going to try
    # the basic thing first and flatten the audio by it's temporal envelope

    # Convert to mono if stereo
    # if len(audio.shape) > 1:
    #     audio = np.mean(audio, axis=1)
    # breakpoint()

    # Normalize audio
    audio = audio / torch.max(torch.abs(audio))

    envelope, smooth, resamp = extract_envelope(audio)

    # divide audio by envelope - should flatten?
    flattened_audio = audio / torch.from_numpy(envelope).to(audio.device)

    return audio, flattened_audio


def compress(audio, sr=44100):
    # audio input shape: [batch, samples]
    # audio output shape: [batch, samples]
    audio = audio.unsqueeze(1)
    d = audio.device

    # repeat val along dimensions
    def param(val):
        return torch.tensor([val]).to(d).repeat(audio.shape[0])

    # breakpoint()
    audio = compressor(audio,
               sample_rate=sr,
               threshold_db=param(-60.0),
               ratio=param(float('inf')),
               attack_ms=param(10),
               release_ms=param(100.0),
               knee_db=param(0.01),
               makeup_gain_db=param(0.0),
               lookahead_samples=int(sr*1e-3)
               # lookahead_samples=int(sr*1e1)
            )

    # normalize audio
    # breakpoint()
    audio = audio.squeeze(1)
    peak = audio.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    audio = audio / peak


    return audio

def extract_envelope_np(wv, target_length=64, cutoff_freq=30):
    """
    Extract a fixed-length temporal envelope from numpy arr

    Parameters:
    -----------
    wv : np.arr
        Audio

    target_length : int
        Desired length of the output envelope (default: 2048)

    cutoff_freq : float
        Cutoff frequency for the lowpass filter in Hz (default: 30)


    Returns:
    --------
    tuple
        (original_envelope, smoothed_envelope, resampled_envelope)
    """
    # sample_rate, audio = wavfile.read(filepath)
    audio = wv

    # Convert to mono if stereo (shouldn't need)
    # if len(audio.shape) > 1:
    #     audio = np.mean(audio, axis=1)

    # Normalize audio
    # audio = audio.astype(float) / np.max(np.abs(audio))

    # Calculate temporal envelope using Hilbert transform
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)

    # assume 44.1kHz sample rate (this won't matter too much in reality as:
    #  normalized_cutoff = 30 / 22050 = 0.00136
    #  normalized_cutoff = 30 / 24000 = 0.00125
    # this is only an 8.8% difference in cutoff frequency from 44.1k vs 48k
    sample_rate = 44100

    # Design and apply lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normalized_cutoff, btype='low')
    smoothed_envelope = filtfilt(b, a, envelope)

    # the nyquist frequency for 44100 is (86.13/2) = 43
    #                       for 48000 is (93.75/2) = 47
    # so I'm not too worried about aliasing


    # # Resample to target length
    # batch, samps = smoothed_envelope.shape

    # original_indices = np.linspace(0, samps - 1, samps)
    # target_indices = np.linspace(0, samps - 1, target_length)

    # resampled_envelope = np.interp(
    #     target_indices,
    #     original_indices,
    #     smoothed_envelope.reshape(-1, samps)
    # ).reshape(batch, target_length)

    # original_indices = np.linspace(0, len(smoothed_envelope)-1, len(smoothed_envelope))
    # target_indices = np.linspace(0, len(smoothed_envelope)-1, target_length)
    # resampled_envelope = np.interp(target_indices, original_indices, smoothed_envelope)

    batch, samps = smoothed_envelope.shape

    original_indices = np.linspace(0, samps - 1, samps)
    target_indices = np.linspace(0, samps - 1, target_length)

    resampled_envelope = np.empty((batch, target_length), dtype=smoothed_envelope.dtype)

    for b in range(batch):
        resampled_envelope[b] = np.interp(
            target_indices,
            original_indices,
            smoothed_envelope[b]
        )

    # we only care about the resampled one for now
    # resampled_envelope = torch.from_numpy(resampled_envelope).to(torch.float32).to(device)
    return envelope, smoothed_envelope, resampled_envelope
