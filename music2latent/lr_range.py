import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .train import Trainer
from .hparams import hparams
from . import misc


class LRRangeTest(Trainer):
    def __init__(
        self,
        min_lr=1e-7,
        max_lr=1,
        num_iters=1000,
        beta=0.98,
    ):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iters = num_iters
        self.beta = beta  # for smoothed loss

        # optimizer (reuse your params)
        # self.optimizer = torch.optim.AdamW(
        #     self.gen.parameters(),
        #     lr=min_lr,
        #     weight_decay=hparams.weight_decay,
        # )
        self.optimizer = torch.optim.RAdam(self.gen.parameters(), lr=min_lr, betas=(hparams.optimizer_beta1, hparams.optimizer_beta2))

        # exponential LR schedule
        self.lr_mult = (max_lr / min_lr) ** (1 / num_iters)

        self.lrs = []
        self.losses = []

    def run(self):
        if misc.get_rank() != 0:
            return  # only run on rank 0

        avg_loss = 0.0
        best_loss = float("inf")

        self.gen.train()

        data_iter = iter(self.dl)

        for step in tqdm(range(self.num_iters), desc="LR range test"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dl)
                batch = next(data_iter)

            # print a hash / id from batch
            # print(batch[0][:5])  # or batch.shape, indices, filenames, etc

            loss = self.train_step(batch)


            # smooth the loss (???)
            avg_loss = self.beta * avg_loss + (1 - self.beta) * loss
            smoothed_loss = avg_loss / (1 - self.beta ** (step + 1))
            smoothed_loss = loss # don't smooth the loss

            lr = self.optimizer.param_groups[0]["lr"]

            self.lrs.append(lr)
            self.losses.append(smoothed_loss)
            print(f'{smoothed_loss}, {lr}')

            # update best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # stop if loss explodes
            if step > 10000 and smoothed_loss > 4 * best_loss:
                print("Stopping early: loss diverged")
                break

            # update LR
            for pg in self.optimizer.param_groups:
                pg["lr"] *= self.lr_mult

        self.save_results()

    # def train_step(self, batch):
    #     self.optimizer.zero_grad(set_to_none=True)

    #     wv = batch.to(self.device)

    #     loss = self.train_it(wv)

    #     # loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(
    #     #     self.gen.parameters(), hparams.grad_clip
    #     # )
    #     # self.optimizer.step()

    #     # return loss.item()
    #     return loss

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)

        wv = batch.to(self.device)

        with torch.cuda.amp.autocast(False):
            loss = self.forward_loss(wv)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in LR test")

        loss.backward()
        self.optimizer.step()

        return loss.item()


    def save_results(self):
        save_dir = os.path.join(self.save_path, "lr_range_test")
        os.makedirs(save_dir, exist_ok=True)

        # save raw data
        np.save(os.path.join(save_dir, "lrs.npy"), np.array(self.lrs))
        np.save(os.path.join(save_dir, "losses.npy"), np.array(self.losses))

        # plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("LR Range Test")
        plt.grid(True)

        plt.savefig(os.path.join(save_dir, "lr_range_test.png"))
        plt.close()

        print(f"LR range test saved to {save_dir}")


def main():
    tester = LRRangeTest(
        min_lr=1e-7,
        # min_lr=1e-5,
        max_lr=1e-1,
        num_iters=100,
    )
    tester.run()
