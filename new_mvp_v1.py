import argparse
import json
import os
import rich

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, DeviceStatsMonitor, Timer
from pytorch_lightning.loggers import TensorBoardLogger


# =============================================================================
# CONFIGURATION
# =============================================================================
# Enable cuDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True
# Set float32 matmul precision for faster training
torch.set_float32_matmul_precision('high')


# ==============================================================================
# 1. DATASET: Stereo inputs + full complex STFT + hierarchical class weights
# ==============================================================================
class QualityOptimizedAudioConfigDataset(Dataset):
    """
    Loads .WAV/.JSON pairs, extracts left/right mel spectrograms + full complex STFT,
    and computes inverse-frequency class weights for each hierarchical level.
    """
    def __init__(
        self,
        data_directory: str,
        filter_hierarchy_mapping: dict,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        frequency_min_hz: float = 8.0,
        frequency_max_hz: float = 22050.0,
        use_full_complex_stft: bool = True,
        compute_class_weights: bool = True
    ):
        """
        :param data_directory: Root folder with .wav and .json pairs.
        :param filter_hierarchy_mapping: Dict mapping filter name -> (top_idx, sub_idx, specific_idx)
        :param sample_rate: 48000 Hz per spec.
        :param n_fft: 2048 gives ~23.4 Hz frequency resolution. High resolution captures steep filter slopes.
        :param hop_length: 512 gives ~10.6 ms temporal resolution. ~104 frames for 1.1s audio.
        :param n_mels: 128 mel bands. Standard perceptual compression.
        :param frequency_min_hz: 8.0 Hz (Serum floor)
        :param frequency_max_hz: 22050.0 Hz (Serum ceiling)
        :param use_full_complex_stft: If True, adds real/imag STFT channels (input_channels=6). 
                                      If False, uses left/right mel only (input_channels=2).
        :param compute_class_weights: If True, computes inverse-frequency weights for CrossEntropyLoss.
        """
        super().__init__()
        self.data_directory = data_directory
        self.data_path = Path(data_directory)
        self.filter_hierarchy_mapping = filter_hierarchy_mapping
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frequency_min_hz = frequency_min_hz
        self.frequency_max_hz = frequency_max_hz
        self.use_full_complex_stft = use_full_complex_stft
        self.compute_class_weights = compute_class_weights
        
        # Log bounds for frequency normalization
        self.log_freq_min = np.log10(self.frequency_min_hz)
        self.log_freq_max = np.log10(self.frequency_max_hz)
        self.log_freq_range = self.log_freq_max - self.log_freq_min
        
        # Mel spectrogram transform (operates on mono or stereo input)
        self.mel_spectrogram_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate / 2.0
        )
        
        # Complex spectrogram transform for full phase capture
        # `return_complex` argument is now deprecated and is not effective.
        # `torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with complex dtype.
        self.complex_spectrogram_transform = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None
        )
        
        # Collect valid pairs
        self.file_pairs = []
        print(f"Searching for .WAV + .JSON file pairs under {self.data_directory} ...")
        for wav_path in self.data_path.rglob("*.wav"):
            json_path = wav_path.with_stem(f"{wav_path.stem}_params").with_suffix('.json')
            if json_path.exists():
                self.file_pairs.append({
                    "wav_path": wav_path,
                    "json_path": json_path
                })
        print(f"Found {len(self.file_pairs)} valid audio-config pairs under {self.data_directory}")
        
        self.class_weights = {}
        if compute_class_weights:
            self._compute_hierarchical_class_weights()

    def _compute_hierarchical_class_weights(self):
        """
        Computes inverse-frequency weights for each hierarchical level.
        Formula: weight_i = N_total / (N_classes * count_i)
        Normalizes so weights are roughly in [0.1, 5.0] range to prevent gradient explosion.
        """
        print('Computing hierarchical class weights for filter_1_type...')
        top_counts = np.zeros(5)
        sub_counts = np.zeros(25) # Padded to max observed sub-category count
        specific_counts = np.zeros(108)
        
        total_number_of_pairs = len(self.file_pairs)

        for pair_index, pair_info in enumerate(tqdm(self.file_pairs, desc='Computing hierarchical class weights for all .WAV + .JSON file pairs in the raw dataset')):
            #if pair_index % 1000 == 0:
            #    print(f"DEBUG: Analyzing pair {pair_index} of {total_number_of_pairs}...")
            #print(f"DEBUG: Analyzing {pair_info['json_path']}...")
            with open(pair_info["json_path"], "r") as json_file:
                config_data = json.load(json_file)
            filter_name = config_data["filter_1_type"]
            top_idx, sub_idx, specific_idx = self.filter_hierarchy_mapping[filter_name]
            top_counts[top_idx] += 1
            sub_counts[sub_idx] += 1
            specific_counts[specific_idx] += 1
            
        # Avoid division by zero for unseen classes
        top_counts = np.maximum(top_counts, 1.0)
        sub_counts = np.maximum(sub_counts, 1.0)
        specific_counts = np.maximum(specific_counts, 1.0)
        
        # Inverse frequency weighting
        top_counts = 1.0 / top_counts
        sub_counts = 1.0 / sub_counts
        specific_counts = 1.0 / specific_counts
        
        # Normalize to prevent extreme values during training
        self.class_weights["top"] = torch.tensor(top_counts / top_counts.mean(), dtype=torch.float32)
        self.class_weights["sub"] = torch.tensor(sub_counts / sub_counts.mean(), dtype=torch.float32)
        self.class_weights["specific"] = torch.tensor(specific_counts / specific_counts.mean(), dtype=torch.float32)
        
        print("Computed hierarchical class weights. Top range:", self.class_weights["top"].min().item(), "-", self.class_weights["top"].max().item())

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index: int):
        pair_info = self.file_pairs[index]
        
        # Load audio with soundfile as requested. Returns (num_channels, num_samples) float32 array.
        audio_array, _ = sf.read(pair_info["wav_path"],dtype='float32')
        audio_tensor = torch.from_numpy(audio_array).float().t()
        
        # Split into left and right channels for stereo width modulation capture
        left_channel_audio_tensor = audio_tensor[0:1, :]
        right_channel_audio_tensor = audio_tensor[1:2, :] if audio_tensor.shape[0] > 1 else audio_tensor[0:1, :]
        
        # left_channel_audio_tensor.shape = torch.Size([1, 52800])
        #                                         Channels, N Mels, Time
        # mel_left.shape                  = torch.Size([1, 128, 104])
        #                                         Channels, Freq, Time
        # left_complex_stft.shape         = torch.Size([1, 1025, 104])
        # stft_real_left.shape            = torch.Size([1, 1025, 104])
        # stft_imag_left.shape            = torch.Size([1, 1025, 104])
        
        # Compute mel spectrograms independently for left and right
        mel_left = self.mel_spectrogram_transform(left_channel_audio_tensor)
        mel_right = self.mel_spectrogram_transform(right_channel_audio_tensor)
        
        # Concatenate along channel dimension: [mel_left, mel_right] -> (2, n_mels, time_frames)
        #mel_stereo = torch.cat([mel_left, mel_right], dim=1)
        
        # Extract full complex STFT separately for each channel to preserve explicit inter-channel phase relationships
        # torchaudio.transforms.Spectrogram returns shape (1, freq_bins, time_frames) for mono input
        left_complex_stft = self.complex_spectrogram_transform(left_channel_audio_tensor)
        right_complex_stft = self.complex_spectrogram_transform(right_channel_audio_tensor)
        
        # Separate real and imaginary components for each channel
        stft_real_left = torch.real(left_complex_stft)
        stft_imag_left = torch.imag(left_complex_stft)
        stft_real_right = torch.real(right_complex_stft)
        stft_imag_right = torch.imag(right_complex_stft)
        
        # Concatenate along the channel dimension (dim=1 in torchaudio's (C, F, T) convention)
        # Resulting shape: (6, n_mels + 4*freq_bins, time_frames)
        # This gives the CNN explicit access to: [mel_L, mel_R, real_phase_L, imag_phase_L, real_phase_R, imag_phase_R]
        if self.use_full_complex_stft:
            combined_spectrogram = torch.cat([
                mel_left, 
                mel_right,
                stft_real_left, 
                stft_imag_left,
                stft_real_right, 
                stft_imag_right
            ], dim=1)
        else:
            # Fallback to mel-only if phase features are disabled
            combined_spectrogram = torch.cat([mel_left, mel_right], dim=1)
            
        # Load JSON config
        with open(str(pair_info["json_path"]), "r") as json_file:
            config_data = json.load(json_file)
            
        filter_name = config_data["filter_1_type"]
        frequency_hz = float(config_data["filter_1_freq_hz"])
        
        # Map to hierarchical indices
        top_idx, sub_idx, specific_idx = self.filter_hierarchy_mapping[filter_name]
        
        # Normalize frequency using log scale to [0, 1]
        log_freq = np.log10(frequency_hz)
        normalized_freq = (log_freq - self.log_freq_min) / self.log_freq_range
        normalized_freq = np.clip(normalized_freq, 0.0, 1.0)
        
        return {
            "combined_spectrogram": combined_spectrogram.float(),
            "top_category_idx": torch.tensor(top_idx, dtype=torch.long),
            "sub_category_idx": torch.tensor(sub_idx, dtype=torch.long),
            "specific_filter_idx": torch.tensor(specific_idx, dtype=torch.long),
            "normalized_frequency": torch.tensor(normalized_freq, dtype=torch.float32),
            "actual_frequency_hz": torch.tensor(frequency_hz, dtype=torch.float32)
        }


# ==============================================================================
# 2. MODEL: 4-channel input backbone + hierarchical heads
# ==============================================================================
class PhaseAwareHierarchicalPredictor(torch.nn.Module):
    """
    Multi-task architecture optimized for accuracy. Handles 4-channel inputs 
    (left mel, right mel, STFT real, STFT imag) via expanded initial conv layer.
    """
    def __init__(
        self,
        num_top_categories: int = 5,
        num_sub_categories: int = 25,
        num_specific_filters: int = 108,
        input_channels: int = 1,
        mel_height: int = 128,
        mel_width: int = 104
    ):
        super().__init__()
        
        # Shared Convolutional Backbone
        # First layer expanded to 64 channels to handle richer 4-channel input without losing representational capacity.
        # 2-layer CNN with batch norm and max pooling. Designed for accuracy, not speed.
        self.shared_convolutional_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.shared_linear_projection = torch.nn.Linear(128, 512) # Increased from 256 to 512 for higher capacity
        torch.nn.init.kaiming_normal_(self.shared_linear_projection.weight, nonlinearity='relu')
        
        # Hierarchical Classification Heads
        self.top_category_head = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, num_top_categories)
        )
        
        self.sub_category_head = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, num_sub_categories)
        )
        
        self.specific_filter_head = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_specific_filters)
        )
        
        # Regression Head
        self.frequency_regression_head = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, combined_spectrogram: torch.Tensor):
        shared_features = self.shared_convolutional_backbone(combined_spectrogram)
        pooled_features = self.global_average_pooling(shared_features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        projected_features = torch.relu(self.shared_linear_projection(flattened_features))
        
        return {
            "top_category_logits": self.top_category_head(projected_features),
            "sub_category_logits": self.sub_category_head(projected_features),
            "specific_filter_logits": self.specific_filter_head(projected_features),
            "normalized_frequency_prediction": self.frequency_regression_head(projected_features)
        }


# ==============================================================================
# 3. LIGHTNING MODULE: Weighted hierarchical loss + logarithmic perceptual weighting
# ==============================================================================
class HierarchicalFilterPredictorLightningModule(LightningModule):
    """
    PyTorch Lightning wrapper for hierarchical classification and logarithmic perceptual regression.
    """
    def __init__(self, learning_rate: float = 1e-3, 
                 top_loss_weight: float = 2.0, sub_loss_weight: float = 1.5, 
                 specific_loss_weight: float = 1.0, regression_loss_weight: float = 0.8,
                 huber_delta_hz: float = 15.0,
                 class_weights_top: torch.Tensor = None,
                 class_weights_sub: torch.Tensor = None,
                 class_weights_specific: torch.Tensor = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = PhaseAwareHierarchicalPredictor(input_channels=1)
        
        # CrossEntropyLoss with class imbalance weighting
        self.top_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_top)
        self.sub_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_sub)
        self.specific_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_specific)
        
        # Logarithmic perceptual weighted Huber loss
        self.perceptual_huber_loss_fn = PerceptualWeightedHuberLoss(
            delta_hz=self.hparams.huber_delta_hz,
            perceptual_weight_func=self.compute_perceptual_weight
        )
        
        self.scheduler = None

    def compute_perceptual_weight(self, frequency_hz_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes frequency-dependent loss weight using logarithmic decay.
        Formula: weight = 1.0 / log(frequency_hz + 1.0)
        Rationale: Low-frequency cutoff errors are perceptually salient, but a linear or inverse 
        weight explodes at very low frequencies. Logarithmic decay provides a smooth, bounded 
        penalty that matches musical pitch discrimination curves without destabilizing gradients.
        """
        epsilon = 1.0
        weight = 1.0 / torch.log(torch.abs(frequency_hz_tensor) + epsilon)
        # Normalize to keep loss scale consistent across batches
        return weight / weight.mean()

    def forward(self, combined_spectrogram: torch.Tensor):
        return self.model(combined_spectrogram)

    def training_step(self, batch: dict, batch_index: int):
        combined_spectrogram = batch["combined_spectrogram"]
        predictions = self(combined_spectrogram)
        
        # Classification losses with class imbalance weighting
        top_loss = self.top_loss_fn(predictions["top_category_logits"], batch["top_category_idx"])
        sub_loss = self.sub_loss_fn(predictions["sub_category_logits"], batch["sub_category_idx"])
        specific_loss = self.specific_loss_fn(predictions["specific_filter_logits"], batch["specific_filter_idx"])
        
        # Regression loss with logarithmic perceptual weighting
        regression_loss = self.perceptual_huber_loss_fn(
            predictions["normalized_frequency_prediction"],
            batch["normalized_frequency"],
            batch["actual_frequency_hz"]
        )
        
        # Combined loss with hierarchical weights
        total_loss = (self.hparams.top_loss_weight * top_loss) + \
                     (self.hparams.sub_loss_weight * sub_loss) + \
                     (self.hparams.specific_loss_weight * specific_loss) + \
                     (self.hparams.regression_loss_weight * regression_loss)
        
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_top_loss", top_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_sub_loss", sub_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_specific_loss", specific_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_regression_loss", regression_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch: dict, batch_index: int):
        combined_spectrogram = batch["combined_spectrogram"]
        predictions = self(combined_spectrogram)
        
        top_loss = self.top_loss_fn(predictions["top_category_logits"], batch["top_category_idx"])
        sub_loss = self.sub_loss_fn(predictions["sub_category_logits"], batch["sub_category_idx"])
        specific_loss = self.specific_loss_fn(predictions["specific_filter_logits"], batch["specific_filter_idx"])
        regression_loss = self.perceptual_huber_loss_fn(
            predictions["normalized_frequency_prediction"],
            batch["normalized_frequency"],
            batch["actual_frequency_hz"]
        )
        
        total_loss = (self.hparams.top_loss_weight * top_loss) + \
                     (self.hparams.sub_loss_weight * sub_loss) + \
                     (self.hparams.specific_loss_weight * specific_loss) + \
                     (self.hparams.regression_loss_weight * regression_loss)
        
        self.log("val_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_top_loss", top_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_sub_loss", sub_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_specific_loss", specific_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_regression_loss", regression_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Hierarchical accuracy metrics
        top_accuracy = (torch.argmax(predictions["top_category_logits"], dim=1) == batch["top_category_idx"]).float().mean()
        sub_accuracy = (torch.argmax(predictions["sub_category_logits"], dim=1) == batch["sub_category_idx"]).float().mean()
        specific_accuracy = (torch.argmax(predictions["specific_filter_logits"], dim=1) == batch["specific_filter_idx"]).float().mean()
        
        self.log("val_top_accuracy", top_accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_sub_accuracy", sub_accuracy, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_specific_accuracy", specific_accuracy, prog_bar=True, on_step=True, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        
        # Keep this even if not used, because this makes it easy to switch out optimizers for testing!
        # CAWR prevents the learning rate from decaying to near-zero too early, allowing the model to escape local minima.
        # eta_min=1e-6 ensures gradients remain non-zero for fine-tuning late in training.
        scheduler_cawr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # Warmup epochs
            T_mult=2,    # Expand cycle length
            eta_min=1e-6 # Minimum learning rate floor
        )
        
        # Keep this even if not used, because this makes it easy to switch out optimizers for testing!
        scheduler_rlop = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Halve LR when plateau detected
            patience=10, # Wait 10 epochs before reducing
            min_lr=1e-6  # Minimum learning rate floor
        )
        
        scheduler = scheduler_cawr
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            'monitor': 'val_loss'
        }


class PerceptualWeightedHuberLoss(torch.nn.Module):
    """
    Applies frequency-dependent weighting to the Huber loss.
    Matches human auditory perception where low-frequency cutoff errors are more perceptually salient.
    """
    def __init__(self, delta_hz: float = 15.0, perceptual_weight_func=None):
        super().__init__()
        self.huber = torch.nn.SmoothL1Loss(reduction='none')
        self.delta = delta_hz
        self.weight_func = perceptual_weight_func

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, frequency_hz: torch.Tensor):
        raw_loss = self.huber(prediction, target)
        weight = self.weight_func(frequency_hz)
        weighted_loss = raw_loss * weight
        return weighted_loss.mean()


# ==============================================================================
# 4. MAIN EXECUTION: Quality-optimized configuration
# ==============================================================================
def main():
    # Define the exact 3-level hierarchy mapping for all 108 filters
    filter_hierarchy_mapping = {
        'MG Low 6': (0, 0, 0), 'MG Low 12': (0, 0, 1), 'MG Low 18': (0, 0, 2), 'MG Low 24': (0, 0, 3),
        'Low 6': (0, 1, 4), 'Low 12': (0, 1, 5), 'Low 18': (0, 1, 6), 'Low 24': (0, 1, 7),
        'High 6': (0, 2, 8), 'High 12': (0, 2, 9), 'High 18': (0, 2, 10), 'High 24': (0, 2, 11),
        'Band 12': (0, 3, 12), 'Band 24': (0, 3, 13), 'Peak 12': (0, 4, 14), 'Peak 24': (0, 4, 15),
        'Notch 12': (0, 5, 16), 'Notch 24': (0, 5, 17),
        'LH 6': (1, 0, 18), 'LH 12': (1, 0, 19), 'LB 12': (1, 1, 20), 'LP 12': (1, 2, 21),
        'LN 12': (1, 3, 22), 'HB 12': (1, 4, 23), 'HP 12': (1, 5, 24), 'HN 12': (1, 6, 25),
        'BP 12': (1, 7, 26), 'BN 12': (1, 8, 27), 'PP 12': (1, 9, 28), 'PN 12': (1, 10, 29),
        'NN 12': (1, 11, 30), 'L/B/H 12': (1, 12, 31), 'L/B/H 24': (1, 12, 32),
        'L/P/H 12': (1, 13, 33), 'L/P/H 24': (1, 13, 34), 'L/N/H 12': (1, 14, 35), 'L/N/H 24': (1, 14, 36),
        'B/P/N 12': (1, 15, 37), 'B/P/N 24': (1, 15, 38),
        'Cmb +': (2, 0, 39), 'Cmb -': (2, 0, 40), 'Cmb L6+': (2, 0, 41), 'Cmb L6-': (2, 0, 42),
        'Cmb H6+': (2, 0, 43), 'Cmb H6-': (2, 0, 44), 'Cmb HL6+': (2, 0, 45), 'Cmb HL6-': (2, 0, 46),
        'Flg +': (2, 1, 47), 'Flg -': (2, 1, 48), 'Flg L6+': (2, 1, 49), 'Flg L6-': (2, 1, 50),
        'Flg H6+': (2, 1, 51), 'Flg H6-': (2, 1, 52), 'Flg HL6+': (2, 1, 53), 'Flg HL6-': (2, 1, 54),
        'Phs 12+': (2, 2, 55), 'Phs 12-': (2, 2, 56), 'Phs 24+': (2, 2, 57), 'Phs 24-': (2, 2, 58),
        'Phs 36+': (2, 2, 59), 'Phs 36-': (2, 2, 60), 'Phs 48+': (2, 2, 61), 'Phs 48-': (2, 2, 62),
        'Phs 48L6+': (2, 2, 63), 'Phs 48L6-': (2, 2, 64), 'Phs 48H6+': (2, 2, 65), 'Phs 48H6-': (2, 2, 66),
        'Phs 48HL6+': (2, 2, 67), 'Phs 48HL6-': (2, 2, 68), 'FPhs 12HL6+': (2, 3, 69), 'FPhs 12HL6-': (2, 3, 70),
        'Low EQ 6': (3, 0, 71), 'Low EQ 12': (3, 0, 72), 'Band EQ 12': (3, 1, 73),
        'High EQ 6': (3, 2, 74), 'High EQ 12': (3, 2, 75), 'Ring Mod': (3, 3, 76), 'Ring Modx2': (3, 3, 77),
        'SampHold': (3, 4, 78), 'SampHold-': (3, 4, 79), 'Combs': (3, 5, 80), 'Allpasses': (3, 5, 81),
        'Reverb': (3, 6, 82), 'French LP': (3, 7, 83), 'German LP': (3, 7, 84), 'Add Bass': (3, 8, 85),
        'Formant-I': (3, 9, 86), 'Formant-II': (3, 9, 87), 'Formant-III': (3, 9, 88), 'Bandreject': (3, 10, 89),
        'Dist.Comb 1 LP': (3, 11, 90), 'Dist.Comb 1 BP': (3, 11, 91), 'Dist.Comb 2 LP': (3, 11, 92), 'Dist.Comb 2 BP': (3, 11, 93),
        'Scream LP': (3, 12, 94), 'Scream BP': (3, 12, 95), 'Wsp': (4, 0, 96), 'DJ Mixer': (4, 1, 97),
        'Diffusor': (4, 2, 98), 'MG Ladder': (4, 3, 99), 'Acid Ladder': (4, 3, 100), 'EMS Ladder': (4, 3, 101),
        'MG Dirty': (4, 4, 102), 'PZ SVF': (4, 5, 103), 'Comb 2': (4, 6, 104), 'Exp MM': (4, 7, 105),
        'Exp BPF': (4, 8, 106), 'K35': (4, 9, 107)
    }
    
    data_directory_string = "./renders/lpf_mvp/Serum2"
    
    # Quality-optimized dataset: stereo inputs + full complex STFT + class imbalance weighting
    full_dataset = QualityOptimizedAudioConfigDataset(
        data_directory=data_directory_string,
        filter_hierarchy_mapping=filter_hierarchy_mapping,
        sample_rate=48000,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        frequency_min_hz=8.0,
        frequency_max_hz=22050.0,
        use_full_complex_stft=True,
        compute_class_weights=True
    )
    
    # Pass computed class weights to the Lightning module
    lightning_module = HierarchicalFilterPredictorLightningModule(
        learning_rate=1e-3,
        top_loss_weight=2.0,
        sub_loss_weight=1.5,
        specific_loss_weight=1.0,
        regression_loss_weight=0.8,
        huber_delta_hz=15.0,
        class_weights_top=full_dataset.class_weights["top"],
        class_weights_sub=full_dataset.class_weights["sub"],
        class_weights_specific=full_dataset.class_weights["specific"]
    )
    
    # Split into train/validation sets
    val_dataset_length = int(len(full_dataset) * 0.1)
    train_dataset_length = len(full_dataset) - val_dataset_length
    print(f"DEBUG: val_dataset_length: {val_dataset_length} train_dataset_length: {train_dataset_length} based on len(full_dataset) == {len(full_dataset)}")
    
    # Split dataset with fixed seed for reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_dataset_length, val_dataset_length],
        generator=torch.Generator().manual_seed(667)
    )
    print(f"Split: {len(train_dataset)} training, {len(val_dataset)} validation")
    
    # Reduced batch size accommodates higher VRAM usage from 4-channel inputs
    # Reduced batch size from 256 to 64 to trade ~3x slower epoch time for noticeably better generalization.
    # Large batches often converge to sharp minima that perform poorly on unseen audio material.
    # num_workers=6 saturates your CPU cores for parallel STFT computation without blocking the GPU.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  # Preloads 2 batches ahead to keep GPU fed during CPU-heavy STFT computation
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  # Preloads 2 batches ahead to keep GPU fed during CPU-heavy STFT computation
    )
    
    # Add timestamps to logs and callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename=f"new_mvp_v1_{timestamp}_" + "{epoch:04d}_{val_loss:.4g}_{val_specific_loss:.4g}_{val_regression_loss:.4g}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        verbose=True,
        #auto_insert_metric_name=False
    )
    
    device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
    time_stats_monitor = Timer(duration=None, verbose=True)
    rich_model_summary = RichModelSummary(max_depth=-1)
    
    logger = TensorBoardLogger('tb_logs', name=f'lpf_prediction_{timestamp}')
    
    trainer = Trainer(
        max_epochs=150, # Extended training for accuracy optimization
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, device_stats_monitor, time_stats_monitor, rich_model_summary],
        logger=logger,
        default_root_dir="./logs",
        precision="16-mixed",
        log_every_n_steps=50,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        profiler="simple"
    )
    
    trainer.fit(lightning_module, train_dataloader, val_dataloader)
    print("Training complete. Best quality checkpoint saved to ./checkpoints/")
    print(f"Best model: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
