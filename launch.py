# launch.py
import torch
import torch._dynamo
torch._dynamo.config.automatic_dynamic_shapes = False # solves error raised when generating samples with different length
import argparse
from music2latent.config_loader import load_config


if __name__ == "__main__":
    # breakpoint()
    parser = argparse.ArgumentParser(description="Train or run the music2latent model.")
    # parser.add_argument("--config", type=str, default=None, help="Path to a configuration file.")
    args = parser.parse_args()

    config_path = 'configs/config_general_base.py'
    load_config(config_path)

    # if args.config:
    #     load_config(config_path)

    from music2latent.train import main
    # main(args.config)
    main(config_path)
