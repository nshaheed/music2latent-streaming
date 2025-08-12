import os
import argparse
# os.chdir("/data/nils/repos/codecs_benchmark/music2latent")
# os.chdir("/home/groups/brg/nshaheed/music2latent-streaming/")
# os.chdir("/Users/nshaheed/programming/music2latent-streaming")

from music2latent.hparams import hparams
from music2latent import EncoderDecoder
from music2latent.transforms import StreamableSTFT

from music2latent.config_loader import load_config

from music2latent.ema import ExponentialMovingAverage
from music2latent.models_stream import *
import torch
# from IPython.display import display, Audio
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="export the music2latent model.")
parser.add_argument("checkpoint")
parser.add_argument("config")
args = parser.parse_args()

def get_models(load_path_inference = None):
    gen = UNet()
    if load_path_inference is not None:
        checkpoint = torch.load(load_path_inference, map_location="cpu", weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
        # if checkpoint['ema_state_dict'] exists, init ema model and load ema_state_dict
        if 'ema_state_dict' in checkpoint:
            ema = ExponentialMovingAverage(gen.parameters(), decay=hparams.ema_momentum)
            ema.load_state_dict(checkpoint['ema_state_dict'])
            ema.copy_to()
            with ema.average_parameters():
                checkpoint['gen_state_dict'] = gen.state_dict()
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=True)
        # self.gen = torch.jit.script(gen)
    return gen


import nn_tilde
import torch
import cached_conv as cc 

cc.MAX_BATCH_SIZE = 1


class StreamingM2L(nn_tilde.Module):
    def __init__(self, net, transform):
        super().__init__()
        self.net = net
        self.transform = transform 
        self.comp_ratio = 4096
        self.latent_size = 64
        self.sigma_rescale = 0.06
        self.diffusion_steps = 1 
        self.hop = hparams.hop
        self.sigma_min = hparams.sigma_min
        self.sigma_max = hparams.sigma_max
        self.rho = hparams.rho
        self.freq_downsample_list = hparams.freq_downsample_list
        self.mixed_precision = hparams.mixed_precision
            
        self.register_method(
                "encode",
                in_channels=1,
                in_ratio=1,
                out_channels=self.latent_size,
                out_ratio=self.comp_ratio,
                input_labels=['(signal) Audio in'],
                output_labels=[f"latent {i}" for i in range(self.latent_size)],
                test_buffer_size=32768*2,
            )

        self.register_method("decode",
                            in_channels=self.latent_size,
                            in_ratio=self.comp_ratio,
                            out_channels=1,
                            out_ratio=1,
                            test_buffer_size=32768*2,
                            input_labels=[
                                f'(signal) Latent dimension {i+1}'
                                for i in range(self.latent_size)
                            ],
                            output_labels=[
                                '(signal) Audio out'
                            ])
        
    def get_sigma(self, i: int, k: int):
        sigma_min, sigma_max = self.sigma_min, self.sigma_max
        return (sigma_min**(1./self.rho) + ((i-1)/(k-1))*(sigma_max**(1./self.rho)-sigma_min**(1./self.rho)))**self.rho

    def denoise(self, noisy_samples: torch.Tensor, sigma: float, latents: torch.Tensor):
        # Denoise samples
        # with torch.no_grad():
        #     with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        #         print(latents.shape, noisy_samples.shape, sigma)
        pred_samples = self.net.forward_generator(latents, noisy_samples, sigma)
        # Sample noise
        pred_noises = torch.randn_like(pred_samples)
        return pred_noises, pred_samples

    def reverse_step(self, x: torch.Tensor, noise: torch.Tensor, sigma: float):
        return x + ((sigma**2 - self.sigma_min**2)**0.5)*noise

    def reverse_diffusion(self, initial_noise: torch.Tensor, latents: torch.Tensor):
        next_noisy_samples = initial_noise
        # Reverse process step-by-step
        # for k in range(self.diffusion_steps):
        k = 0
        # Get sigma values
        sigma = self.get_sigma(self.diffusion_steps+1-k, self.diffusion_steps+1)
        # next_sigma = self.get_sigma(self.diffusion_steps-k, self.diffusion_steps+1)

        # Denoise 
        noisy_samples = next_noisy_samples
        _, pred_samples = self.denoise(noisy_samples, sigma, latents)

        # Step to next (lower) noise level
        # next_noisy_samples = self.reverse_step(pred_samples, pred_noises, next_sigma)

        return pred_samples
    
    
    def decode_to_representation(self, latents: torch.Tensor):
        latents = latents*self.sigma_rescale
        num_samples = latents.shape[0]
        downscaling_factor = 2**self.freq_downsample_list.count(0)
        sample_length = int(latents.shape[-1]*downscaling_factor)
        initial_noise = torch.randn((num_samples, 2, self.hop*2, sample_length))*self.sigma_max
        
        decoded_spectrograms = self.reverse_diffusion(initial_noise, latents=latents)
        return decoded_spectrograms

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        repr_encoder = self.transform.forward(x[:1])
        latent = self.net.encoder(repr_encoder, extract_features=False)/self.sigma_rescale
        return latent.repeat(n, 1, 1)
    
    
    @torch.jit.export
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n = latent.shape[0]
        repr = self.decode_to_representation(latent[:1])
        waveform = self.transform.inverse(repr)
        return waveform.repeat(n, 1, 1)
                                                                                     
cc.use_cached_conv(True)

# config = "/home/groups/brg/nshaheed/music2latent-streaming/checkpoints/2025-07-24 12:15:19.566459/config.py"
# config = "/Users/nshaheed/Downloads/2025-07-24 12_15_19.566459/config.py"
# config = "/Users/nshaheed/Downloads/config.py"
config = args.config
load_config(config)
# ckpt_path = "/home/groups/brg/nshaheed/music2latent-streaming/checkpoints/2025-07-24 12:15:19.566459/model_fid_1.3627121132819016_loss_100.68_iters_560000.pt"
# ckpt_path = "/Users/nshaheed/Downloads/2025-07-24 12_15_19.566459/model_fid_1.3627121132819016_loss_100.68_iters_560000.pt"
# ckpt_path = "/Users/nshaheed/Downloads/model_fid_1.6845580693723368_loss_106.32_iters_280000.pt"
ckpt_path = args.checkpoint
gen = get_models(ckpt_path)
transform = StreamableSTFT(nfft = hparams.hop*4, hop_size = hparams.hop, skip_features = 1, stream = True)


model = StreamingM2L(net=gen, transform = transform)
