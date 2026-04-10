import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torcheval.metrics import FrechetInceptionDistance
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    DEVICE,
    TRAIN_DIR,
    VAL_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
    SAVE_MODEL,
    SAVE_MODEL_EPOCHS,
    CHECKPOINT,
    SAVE_HISTORY_VAL_DIR,
)
from dataset import MapDataset
from models import Discriminator, Generator
from loss import PerceptualAndStyleLoss
from utils import save_checkpoint, load_checkpoint


class LambdaTuningEnv(gym.Env):
    """Reinforcement Learning environment for tuning loss function weights (lambdas)"""

    def __init__(self):
        super(LambdaTuningEnv, self).__init__()

        # Action space: L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA
        self.action_space = spaces.Box(
            low=np.array([-100.0, -100.0, -100.0]),
            high=np.array([100.0, 100.0, 100.0]),
            dtype=np.float64,
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-100.0] * 8), high=np.array([100.0] * 8), dtype=np.float64
        )

        self.best_image_score = None
        self.current_episode = 1
        self.current_step = 1
        self.overall_step = 1
        self.best_score = None

        self.current_l1_lambda = 0
        self.current_perc_lambda = 0
        self.current_style_lambda = 0
        self.current_l1_loss = 0
        self.current_perc_loss = 0
        self.current_style_loss = 0
        self.current_val_loss = 0

        # Initialize models
        self.disc = Discriminator(in_channels=3).to(DEVICE)
        self.gen = Generator(in_channels=3, features=64).to(DEVICE)
        self.opt_disc = optim.Adam(
            self.disc.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
        )
        self.g_scaler = torch.amp.GradScaler()
        self.d_scaler = torch.amp.GradScaler()

        if False:  # LOAD_MODEL
            load_checkpoint(
                f"{CHECKPOINT_DIR}/gen_{CHECKPOINT}.pth.tar",
                self.gen,
                self.opt_gen,
                LEARNING_RATE,
            )
            load_checkpoint(
                f"{CHECKPOINT_DIR}/disc_{CHECKPOINT}.pth.tar",
                self.disc,
                self.opt_disc,
                LEARNING_RATE,
            )

        self.HISTORY_VAL = {
            "step": [],
            "overall_step": [],
            "episode": [],
            "generator_loss": [],
            "discriminator_loss": [],
            "training_loss": [],
            "validation_loss": [],
            "l1_lambda": [],
            "l1_loss": [],
            "perc_lambda": [],
            "perc_loss": [],
            "style_lambda": [],
            "style_loss": [],
            "fid_score": [],
            "psnr_score": [],
            "ssim_score": [],
            "lpips_score": [],
            "reward": [],
        }

        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()
        self.PERC_STYLE_LOSS = PerceptualAndStyleLoss(
            style_layers=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
            content_layers=["relu2_2", "relu3_3", "relu4_3", "relu5_1", "relu5_3"],
        ).to(DEVICE)

    def reset(self, seed=None, options=None):
        """Reset environment"""
        self.best_score = None
        self.current_step = 1
        print("reset !!!")
        return np.array([1.0] * 3 + [100] * 5), {}

    def step(self, action):
        """Execute one step in the environment"""
        L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA = action

        print("==========================")
        print("Episode: ", self.current_episode)
        print("Step: ", self.current_step)
        print("overall_step: ", self.overall_step)
        print(f"L1_LAMBDA: {L1_LAMBDA}")
        print(f"PERC_LAMBDA: {PERC_LAMBDA}")
        print(f"STYLE_LAMBDA: {STYLE_LAMBDA}")

        (
            val_loss,
            l1_loss,
            perc_loss,
            style_loss,
            L1_LAMBDA,
            PERC_LAMBDA,
            STYLE_LAMBDA,
            score,
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        ) = self.train_and_evaluate(L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA)

        done = False
        image_score = -fid_score
        print("Image Score : ", image_score)

        reward = image_score
        print(f"reward: {reward}")

        self.HISTORY_VAL["reward"].append(reward)
        self.HISTORY_VAL["step"].append(self.current_step)
        self.HISTORY_VAL["overall_step"].append(self.overall_step)
        self.HISTORY_VAL["episode"].append(self.current_episode)
        self.HISTORY_VAL["validation_loss"].append(val_loss)
        self.HISTORY_VAL["l1_loss"].append(l1_loss)
        self.HISTORY_VAL["l1_lambda"].append(L1_LAMBDA)
        self.HISTORY_VAL["perc_loss"].append(perc_loss)
        self.HISTORY_VAL["perc_lambda"].append(PERC_LAMBDA)
        self.HISTORY_VAL["style_loss"].append(style_loss)
        self.HISTORY_VAL["style_lambda"].append(STYLE_LAMBDA)
        self.HISTORY_VAL["fid_score"].append(fid_score)
        self.HISTORY_VAL["psnr_score"].append(psnr_score)
        self.HISTORY_VAL["ssim_score"].append(ssim_score)
        self.HISTORY_VAL["lpips_score"].append(lpips_score)

        if self.best_image_score is None or reward > self.best_image_score:
            self.best_image_score = reward
            print(f"New Best Image Score : {self.best_image_score}")
            save_checkpoint(
                self.gen,
                self.opt_gen,
                filepath=f"{CHECKPOINT_DIR}/best_gen_{CHECKPOINT + self.overall_step}.pth.tar",
            )
            save_checkpoint(
                self.disc,
                self.opt_disc,
                filepath=f"{CHECKPOINT_DIR}/best_disc_{CHECKPOINT + self.overall_step}.pth.tar",
            )

        if (self.current_step >= 5) or (
            self.best_score is not None and self.best_score < reward
        ):
            done = True
            print(f"Episode: {self.current_episode} is done...")
            self.current_episode += 1
            self.best_score = reward

        if self.best_score is None:
            self.best_score = reward

        if SAVE_MODEL and (self.overall_step) % SAVE_MODEL_EPOCHS == 0:
            save_checkpoint(
                self.gen,
                self.opt_gen,
                filepath=f"{CHECKPOINT_DIR}/gen_{CHECKPOINT + self.overall_step}.pth.tar",
            )
            save_checkpoint(
                self.disc,
                self.opt_disc,
                filepath=f"{CHECKPOINT_DIR}/disc_{CHECKPOINT + self.overall_step}.pth.tar",
            )

            history_val_df = pd.DataFrame(self.HISTORY_VAL)
            history_val_df.to_csv(
                f"{SAVE_HISTORY_VAL_DIR}/history_{CHECKPOINT + self.overall_step}.csv",
                index=False,
            )

        self.current_step += 1
        self.overall_step += 1

        return (
            np.array(
                [
                    L1_LAMBDA,
                    PERC_LAMBDA,
                    STYLE_LAMBDA,
                    l1_loss,
                    perc_loss,
                    style_loss,
                    fid_score,
                    lpips_score,
                ]
            ),
            reward,
            done,
            False,
            {"best_image_score": self.best_image_score},
        )

    def train_and_evaluate(self, L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA):
        """Train and evaluate model with given lambdas"""
        NUM_EPOCHS = 1

        train_dataset = MapDataset(root_dir=TRAIN_DIR, masked=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        val_dataset = MapDataset(root_dir=VAL_DIR, masked=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        L1_losses = []
        PERC_losses = []
        STYLE_losses = []

        for epoch in range(NUM_EPOCHS):
            self.train_fn(
                epoch + 1,
                train_loader,
                L1_losses,
                PERC_losses,
                STYLE_losses,
                L1_LAMBDA,
                PERC_LAMBDA,
                STYLE_LAMBDA,
            )

            (
                val_loss,
                l1_loss,
                perc_loss,
                style_loss,
                score,
                fid_score,
                psnr_score,
                ssim_score,
                lpips_score,
            ) = self.validate_fn(
                val_loader,
                L1_LAMBDA,
                PERC_LAMBDA,
                STYLE_LAMBDA,
            )

        return (
            val_loss,
            l1_loss,
            perc_loss,
            style_loss,
            L1_LAMBDA,
            PERC_LAMBDA,
            STYLE_LAMBDA,
            score,
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        )

    def train_fn(
        self,
        epoch,
        loader,
        L1_losses,
        PERC_losses,
        STYLE_losses,
        L1_LAMBDA,
        PERC_LAMBDA,
        STYLE_LAMBDA,
    ):
        """Training loop"""
        loop = tqdm(loader, leave=True)
        sum_l1_loss = 0
        sum_perc_loss = 0
        sum_style_loss = 0
        num_samples = 0

        L1_LAMBDA_CLIPED, PERC_LAMBDA_CLIPED, STYLE_LAMBDA_CLIPED = np.clip(
            [L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA], a_min=0.0001, a_max=100
        )

        for idx, (x, y, z) in enumerate(loop):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Train Discriminator
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
                y_fake = self.gen(x)
                D_real = self.disc(x, y)
                D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
                D_fake = self.disc(x, y_fake.detach())
                D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = D_real_loss + D_fake_loss

            self.opt_disc.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.opt_disc)
            self.d_scaler.update()

            # Train generator
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
                D_fake = self.disc(x, y_fake)
                G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
                L1 = self.L1_LOSS(y_fake, y) * L1_LAMBDA_CLIPED
                PERC, STYLE = self.PERC_STYLE_LOSS(y_fake, y, y)
                PERC = PERC * PERC_LAMBDA_CLIPED
                STYLE = STYLE * STYLE_LAMBDA_CLIPED

                sum_l1_loss += L1.item()
                sum_perc_loss += PERC.item()
                sum_style_loss += STYLE.item()
                num_samples += 1

                G_loss = G_fake_loss + L1 + PERC + STYLE

            self.opt_gen.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.opt_gen)
            self.g_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                    D_loss=D_loss.item(),
                    G_loss=G_loss.item(),
                    L1_loss=L1.item(),
                    PERC_loss=PERC.item(),
                    STYLE_loss=STYLE.item(),
                    L1_weight=L1_LAMBDA,
                    PERC_weight=PERC_LAMBDA,
                    STYLE_weight=STYLE_LAMBDA,
                )

        L1_losses.append(sum_l1_loss / num_samples)
        PERC_losses.append(sum_perc_loss / num_samples)
        STYLE_losses.append(sum_style_loss / num_samples)

        self.HISTORY_VAL["generator_loss"].append(G_fake_loss.item())
        self.HISTORY_VAL["discriminator_loss"].append(D_loss.item())
        self.HISTORY_VAL["training_loss"].append(G_loss.item())

    def validate_fn(
        self,
        val_loader,
        L1_LAMBDA,
        PERC_LAMBDA,
        STYLE_LAMBDA,
    ):
        """Validation loop"""
        total_L1, total_PERC, total_STYLE, total_G_loss = 0, 0, 0, 0
        num_samples = 0

        print("Save model to Validate...")
        save_checkpoint(
            self.gen,
            self.opt_gen,
            filepath=f"{CHECKPOINT_DIR}/validate_gen.pth.tar",
        )
        save_checkpoint(
            self.disc,
            self.opt_disc,
            filepath=f"{CHECKPOINT_DIR}/validate_disc.pth.tar",
        )

        validate_disc = Discriminator(in_channels=3).to(DEVICE)
        validate_gen = Generator(in_channels=3, features=64).to(DEVICE)
        validate_opt_disc = optim.Adam(
            validate_disc.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999),
        )
        validate_opt_gen = optim.Adam(
            validate_gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
        )

        print("Load model to Validate...")
        load_checkpoint(
            f"{CHECKPOINT_DIR}/validate_gen.pth.tar",
            validate_gen,
            validate_opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            f"{CHECKPOINT_DIR}/validate_disc.pth.tar",
            validate_disc,
            validate_opt_disc,
            LEARNING_RATE,
        )

        real_list = []
        result_list = []
        resize = transforms.Resize(
            (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
        )

        L1_LAMBDA_CLIPED, PERC_LAMBDA_CLIPED, STYLE_LAMBDA_CLIPED = np.clip(
            [L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA], a_min=0.0001, a_max=100
        )

        validate_gen.eval()
        with torch.no_grad():
            for x, y, z in val_loader:
                x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
                y_fake = validate_gen(x)

                G_fake_loss = self.BCE(
                    validate_disc(x, y_fake), torch.ones_like(validate_disc(x, y_fake))
                )
                L1 = self.L1_LOSS(y_fake, y) * L1_LAMBDA_CLIPED
                PERC, STYLE = self.PERC_STYLE_LOSS(y_fake, y, y)
                PERC = PERC * PERC_LAMBDA_CLIPED
                STYLE = STYLE * STYLE_LAMBDA_CLIPED

                G_loss = G_fake_loss + L1 + PERC + STYLE

                total_L1 += L1.item()
                total_PERC += PERC.item()
                total_STYLE += STYLE.item()
                total_G_loss += G_loss.item()

                num_samples += 1

                y_fake = y_fake * 0.5 + 0.5
                x = x * 0.5 + 0.5
                y = y * 0.5 + 0.5
                z = z.permute(0, 3, 1, 2)

                z_mask = (z > 0.5).to(dtype=torch.float32, device=DEVICE)
                inpainted = (
                    (x * 2 - 1.0) * (1.0 - z_mask) + (y_fake * 2 - 1.0) * z_mask
                ) * 0.5 + 0.5

                result = resize(inpainted)
                real = resize(y)

                result_list.append(result)
                real_list.append(real)

        avg_L1 = total_L1 / num_samples
        avg_PERC = total_PERC / num_samples
        avg_STYLE = total_STYLE / num_samples
        avg_G_loss = total_G_loss / num_samples

        print(
            f"Validation Loss -> G_loss: {avg_G_loss:.6f}, L1: {avg_L1:.6f}, L1_weight: {L1_LAMBDA}, PERC: {avg_PERC:.6f}, PERC_weight: {PERC_LAMBDA}, STYLE: {avg_STYLE:.6f}, STYLE_weight: {STYLE_LAMBDA}"
        )

        score, fid_score, psnr_score, ssim_score, lpips_score = self.calculateScore(
            result_list, real_list
        )

        return (
            avg_G_loss,
            avg_L1,
            avg_PERC,
            avg_STYLE,
            score,
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        )

    def calculateScore(self, result_list, real_list):
        """Calculate evaluation metrics"""
        result_images = torch.cat(result_list, dim=0)
        real_images = torch.cat(real_list, dim=0)

        result_images = (result_images - result_images.min()) / (
            result_images.max() - result_images.min()
        )
        real_images = (real_images - real_images.min()) / (
            real_images.max() - real_images.min()
        )

        try:
            fid_score = self.calculateFID(real_images, result_images)
            psnr_score = self.calculatePSNR(real_images, result_images)
            ssim_score = self.calculateSSIM(real_images, result_images)
            lpips_score = self.calculateLPIPS(real_images, result_images)

            score = 0.5 * psnr_score - 0.5 * fid_score
            score = score.item()
        except:
            score = -100

        return (
            score,
            fid_score.item(),
            psnr_score.item(),
            ssim_score.item() * 10,
            lpips_score.item() * 100,
        )

    def calculateFID(self, real_images, result_images):
        """Calculate FID score"""
        print("Calculating FID...")
        fid = FrechetInceptionDistance()
        fid.update(real_images, True)
        fid.update(result_images, False)
        fid_score = fid.compute()
        return fid_score

    def calculatePSNR(self, img1, img2, max_val=1.0):
        """Calculate PSNR score"""
        print("Calculating PSNR...")
        mse = F.mse_loss(img1, img2)
        psnr = 10 * torch.log10(max_val**2 / mse)
        return psnr

    def calculateSSIM(self, real_images, result_images):
        """Calculate SSIM score"""
        print("Calculating SSIM...")
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        ssim_scores = ssim(real_images, result_images)
        return ssim_scores

    def calculateLPIPS(self, real_images, result_images):
        """Calculate LPIPS score"""
        print("Calculating LPIPS...")
        result_images = result_images * 2 - 1
        real_images = real_images * 2 - 1

        loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_score = loss_fn(real_images, result_images).mean()
        return lpips_score


class EarlyStopCallback(BaseCallback):
    """Early stopping callback for RL training"""

    def __init__(self, patience=500, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.best_mean_fid = float("inf")
        self.steps_since_improvement = 0

    def _on_step(self) -> bool:
        if self.locals.get("infos") is not None:
            image_score = [
                info.get("best_image_score", 100) for info in self.locals["infos"]
            ]

            if len(image_score) > 0:
                mean_image_score = sum(image_score) / len(image_score)
                if mean_image_score < self.best_mean_fid:
                    self.best_mean_image_score = mean_image_score
                    self.steps_since_improvement = 0
                else:
                    self.steps_since_improvement += 1
                    print("Best Image Score : ", self.best_mean_image_score)
                    print("Image Score not improve : ", self.steps_since_improvement)

        if self.steps_since_improvement >= self.patience:
            print(f"⏹️ Early stopping: no improvement in {self.patience} steps")
            return False
        return True


class VecNormalizeHistory:
    """Wrapper to store history of observations and rewards during RL training"""

    def __init__(self, venv):
        self.venv = venv
        self.obs_history = []
        self.reward_history = []

    def step_wait(self):
        """Record step history"""
        obs, rewards, dones, infos = self.venv.step_wait()
        self.obs_history.append(obs.copy())
        self.reward_history.append(rewards.copy())
        return obs, rewards, dones, infos

    def reset(self):
        """Reset and clear history"""
        obs = self.venv.reset()
        self.obs_history = []
        self.reward_history = []
        return obs
