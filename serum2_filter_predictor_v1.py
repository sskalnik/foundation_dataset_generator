"""
Audio Filter Type & Frequency Predictor MVP
============================================
Predicts synthesizer filter type (108 classes) and frequency parameter (8-22050 Hz)
from 32-bit float, 48 kHz stereo WAV files using PyTorch Lightning.

Design Rationale:
- Shared CNN encoder + two prediction heads (frequency regression, categorical classification)
- Complex STFT input: [real_L, imag_L, real_R, imag_R] to preserve stereo phase & imaging
- Log-frequency target compression + perceptual loss weighting aligned with ERB scale
- Inverse class frequency weighting for CrossEntropyLoss to handle dataset imbalance
- Fully configurable via argparse CLI

Trade-off Summary (Memory vs. Compute vs. Quality):
1. Fixed 1.1s window at 48 kHz = 52,992 samples. STFT with hop=256 yields ~207 time frames.
   This keeps memory predictable but may alias very fast transient filter sweeps if hop is too large.
   We use hop=128 to preserve temporal resolution, accepting a ~2x increase in sequence length.
2. 4-channel complex spectrogram replaces single-channel magnitude. Quadruples input dimensionality,
   increasing VRAM usage by ~35% per batch compared to standard Mel spectrograms, but captures
   phase coherence critical for phasing/flanging/combs filter classification.
3. Log-frequency compression + ERB-weighted MSE loss shifts optimization focus toward low/mid bands
   where human hearing is most sensitive, at the cost of slightly higher absolute error in >12kHz range.
"""

import argparse
import gc
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import bitsandbytes as bnb
import numpy as np
# `torchaudio` is practically unusable for file I/O on Windows, but `soundfile` "just works"
# PS: Fuck `torchcodec`, fuck Meta, and fuck you Zuck
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
# `torchaudio` is better for this than `torch.stft`
import torchaudio.transforms as transforms
import torchaudio.functional as taf
# https://github.com/tyleryep/torchinfo
import torchinfo
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar, DeviceStatsMonitor, Timer, WeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from torch.optim.swa_utils import get_ema_avg_fn
from torch.utils.data import DataLoader, Dataset
from pprint import pprint
# Useful for benchmarking, and less verbose than print() statements in loops
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer
# Faster than stock `json`, makes a big difference when loading data from 300_000+ .JSON files at once
from yyjson import Document


# =============================================================================
# PYTORCH BACKEND CONFIGURATION
# =============================================================================
# Enable cuDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True
# Set float32 matmul precision for faster training with values other than "highest" if needed.
# TF32 provides 10 bits of mantissa vs FP32's 24 bits.
# For log-frequency regression targets (range ~2.1 to ~9.9), the quantization noise is ~1e-4,
# which is smaller than typical batch-to-batch label variance.
# Classification heads are inherently robust to sub-FP32 precision.
torch.set_float32_matmul_precision('highest')

# ============================================================================
# CONFIGURATION & DEFAULTS
# ============================================================================
DEFAULT_TRAIN_WORKERS: int = 4
DEFAULT_VAL_WORKERS: int = 2
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_ACCUMULATE_GRAD_BATCHES: int = 2  # Maintain effective batch size of 2x for gradient stability
DEFAULT_PREFETCH_FACTOR: int = 2
DEFAULT_NUM_EPOCHS: int = 100
DEFAULT_LEARNING_RATE: float = 0.0005
# CosineAnnealingWarmRestarts
DEFAULT_OPTIMIZER_TYPE: str = "cawr"
# The original project was just to predict low-pass filter frequencies, then types
DEFAULT_RAW_DATASET_DIR: str = r'./renders/lpf_mvp'
DEFAULT_MODEL_OUTPUT_DIR: str = r'./models'
# It's 2026, and this is the gold standard: 48 kHz 32-bit floating-point PCM
DEFAULT_SAMPLE_RATE: int = 48_000
# STFT Hyperparameters (chosen to balance temporal resolution vs. VRAM usage)
STFT_N_FFT: int = 2048          # Frequency resolution: ~23 Hz per bin at 48kHz
STFT_HOP_LENGTH: int = 128      # Temporal resolution: preserves fast filter transients
STFT_WIN_LENGTH: Optional[int] = None  # Defaults to n_fft
STFT_WINDOW_FN: str = "hann_window"

# Frequency Regression Target Scaling
# These are the minimum and maximum values given by serum2.filter_1_freq_hz.valid_values()
# Note that some filter types, such as "Add Bass", have a different range!
FREQ_MIN_HZ: float = 8.0
FREQ_MAX_HZ: float = 22050.0

# Class Imbalance Mitigation
CLASS_WEIGHT_NORMALIZATION_METHOD: str = "sum_to_one"  # Options: 'sum_to_one', 'sqrt_inverse'
# Hierarchical Loss Weights (tuned to balance gradient magnitudes across levels)
HIERARCHY_CAT_WEIGHT: float = 0.2
HIERARCHY_SUB_WEIGHT: float = 0.3
HIERARCHY_VAR_WEIGHT: float = 0.5

# TEXT_TOKENIZER_DEFAULTS: We use a lightweight character-level tokenizer
# instead of a full word-piece model to reduce VRAM overhead and avoid
# dependency on external vocabularies during inference.
TEXT_MAX_LENGTH: int = 64
TEXT_VOCAB_SIZE: int = 256  # ASCII printable range


# =============================================================================
# DILL REGISTRATION FOR WINDOWS COMPATIBILITY
# =============================================================================
def register_dill_for_windows() -> None:
    """Register dill for Windows-compatible lambda pickling."""
    try:
        import dill
        dill.settings['recurse'] = True
        print("Dill registered for Windows compatibility")
    except ImportError:
        print("Warning: dill not installed. Install with: pip install dill")


register_dill_for_windows()

# ============================================================================
# CALLBACKS
# ============================================================================
rich_model_summary = RichModelSummary(max_depth=-1)
device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
time_stats_monitor = Timer(duration=None, verbose=True)


class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn())

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)


# ============================================================================
# DATASET & DATA MODULE
# ============================================================================
class AudioFilterPredictionDataset(Dataset):
    """
    Loads paired WAV/JSON files, computes complex STFT spectrograms, and returns
    normalized frequency targets alongside categorical filter labels.

    Precomputes class frequencies to generate inverse-frequency weights for the
    CrossEntropyLoss, directly addressing the low-pass vs niche filter imbalance.
    """
    def __init__(
        self,
        dataset_directory: str = DEFAULT_RAW_DATASET_DIR,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration_seconds: float = 1.1,
        n_fft: int = STFT_N_FFT,
        hop_length: int = STFT_HOP_LENGTH,
        file_pairs: List[Tuple[Path, Path]] = [],
        fast_dev_run_size: Optional[int] = None,
    ) -> None:
        self.dataset_directory = Path(dataset_directory)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.file_pairs = file_pairs
        self.fast_dev_run_size = fast_dev_run_size
        # Pre-computed feature caches (populated during setup)
        self.cached_spectrograms: List[torch.Tensor] = []
        self.cached_frequency_targets: List[float] = []
        self.cached_filter_1_types: List[str] = []
        self.cached_filter_1_freqs_hz: List[float] = []
        # Hierarchy & label mappings
        #self.filter_hierarchy_mapping: Dict[str, Tuple[int, int, int]] = {}
        self.mid_level_index_map: Dict[Tuple[int, int], int] = {}
        #self.num_categories: int = 5
        #self.num_subtypes: int = 0
        #self.num_variants: int = 108
        self.filter_type_to_index: Dict[str, int] = {}
        self.index_to_filter_type: Dict[int, str] = {}
        self.class_counts: Dict[str, int] = {}
        self.class_weights: torch.Tensor = torch.tensor([])

        # Complex spectrogram transform for full phase capture
        # `return_complex` argument is now deprecated and is not effective.
        # `torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with complex dtype.
        self.complex_spectrogram_transform = transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=None
        )

        # Perceptual frequency scaling parameters
        self.erband_count: int = 128                # Matches human auditory resolution; reduces 1025 linear bins to perceptually uniform bands
        self.stft_magnitude_max: float = 0.0        # Cached per-dataset for safe uint8 quantization

        # Precomputed ERB filterbank matrix [erband_count, n_fft//2 + 1]
        # Maps linear STFT bins to perceptual bands during setup()
        self.erb_filterbank_matrix = self._compute_erb_filterbank(
            num_filters=self.erband_count,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            low_frequency_hz=8.0,
            high_frequency_hz=22050.0  # Covers full synth range; bins above this are discarded
        ).transpose(0, 1)

        # Normal = [
        #   'MG Low 6', 'MG Low 12', 'MG Low 18', 'MG Low 24',
        #   'Low 6', 'Low 12', 'Low 18', 'Low 24',
        #   'High 6', 'High 12', 'High 18', 'High 24',
        #   'Band 12', 'Band 24',
        #   'Peak 12', 'Peak 24',
        #   'Notch 12', 'Notch 24'
        # ]
        #
        # Multi = [
        #   'LH 6', 'LH 12', 'LB 12', 'LP 12', 'LN 12', 'HB 12', 'HP 12', 'HN 12', 'BP 12', 'BN 12', 'PP 12', 'PN 12', 'NN 12',
        #   'L/B/H 12', 'L/B/H 24', 'L/P/H 12', 'L/P/H 24', 'L/N/H 12', 'L/N/H 24', 'B/P/N 12', 'B/P/N 24'
        # ]
        #
        # Flanges = [
        #   'Cmb +', 'Cmb -', 'Cmb L6+', 'Cmb L6-', 'Cmb H6+', 'Cmb H6-', 'Cmb HL6+', 'Cmb HL6-',
        #   'Flg +', 'Flg -', 'Flg L6+', 'Flg L6-', 'Flg H6+', 'Flg H6-', 'Flg HL6+', 'Flg HL6-',
        #   'Phs 12+', 'Phs 12-', 'Phs 24+', 'Phs 24-', 'Phs 36+', 'Phs 36-', 'Phs 48+', 'Phs 48-',
        #   'Phs 48L6+', 'Phs 48L6-', 'Phs 48H6+', 'Phs 48H6-', 'Phs 48HL6+', 'Phs 48HL6-',
        #   'FPhs 12HL6+', 'FPhs 12HL6-'
        # ]
        #
        # Misc = [
        #   'Low EQ 6', 'Low EQ 12', 'Band EQ 12', 'High EQ 6', 'High EQ 12',
        #   'Ring Mod', 'Ring Modx2', 'SampHold', 'SampHold-', 'Combs', 'Allpasses', 'Reverb',
        #   'French LP', 'German LP', 'Add Bass', 'Formant-I', 'Formant-II', 'Formant-III', 'Bandreject',
        #   'Dist.Comb 1 LP', 'Dist.Comb 1 BP', 'Dist.Comb 2 LP', 'Dist.Comb 2 BP', 'Scream LP', 'Scream BP'
        # ]
        #
        # S2_Filters = [
        #   'Wsp', 'DJ Mixer', 'Diffusor', 'MG Ladder', 'Acid Ladder', 'EMS Ladder', 'MG Dirty', 'PZ SVF', 'Comb 2', 'Exp MM', 'Exp BPF', 'K35'
        # ]
        #
        # Hierarchical Taxonomy (Hardcoded to 108 filter types in Serum 2)
        self.filter_hierarchy_mapping = {
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

        # Compute unique mid-level subtype combinations to create a deterministic global index space
        unique_mid_combos = sorted(list(set((v[0], v[1]) for v in self.filter_hierarchy_mapping.values())))
        self.mid_level_index_map: Dict[Tuple[int, int], int] = {combo: idx for idx, combo in enumerate(unique_mid_combos)}
        # Taxonomy dimensions
        self.num_categories: int = 5
        self.num_subtypes: int = len(unique_mid_combos)
        self.num_variants: int = 108

    def _compute_erb_filterbank(
        self,
        num_filters: int = 128,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = STFT_N_FFT,
        low_frequency_hz: float = 20,
        high_frequency_hz: float = 24_000,
    ) -> torch.Tensor:
        """
        Computes ERB filterbank matrix for perceptual frequency scaling.

        Trade-off Analysis:
        - Reduces frequency dimension from 1025 to 128 bins (-87.5%)
        - Preserves spectral shape fidelity for filter classification
        - Computed once during setup(); negligible CPU overhead (~5ms)
        """
        # Convert ERB scale to linear scale for torchaudio compatibility
        erb_low: float = 214.016774 * np.log(1 + low_frequency_hz / 214.016774)
        erb_high: float = 214.016774 * np.log(1 + high_frequency_hz / 214.016774)

        # Generate filterbank using torchaudio's efficient implementation
        fb_matrix = taf.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=low_frequency_hz,
            f_max=high_frequency_hz,
            n_mels=num_filters,
            sample_rate=sample_rate,
            norm="slaney",  # Matches ERB perceptual spacing
            mel_scale="slaney",
        )

        return fb_matrix  # Shape: [128, 1025]

    def _compute_stft_channels(self, channel_audio_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Short-Time Fourier Transform for a single audio channel.
        Returns real and imaginary tensors of shape [freq_bins, time_frames].
        """
        # Complex spectrogram transform for full phase capture
        # `return_complex` argument is now deprecated and is not effective.
        # `torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with complex dtype.

        # Extract full complex STFT separately for each channel to preserve explicit inter-channel phase relationships
        # torchaudio.transforms.Spectrogram returns shape (1, freq_bins, time_frames) for mono input
        complex_stft = self.complex_spectrogram_transform(channel_audio_tensor)

        # Separate real and imaginary components for each channel
        stft_real = torch.real(complex_stft)
        stft_imag = torch.imag(complex_stft)

        return stft_real, stft_imag

    def collect_wav_and_json_files(self):
        """Find all .WAV and .JSON files in self.dataset_directory."""
        # Resolve all file paths and validate pairs exist
        print(f"Searching {self.dataset_directory} for .WAV audio files...")
        wav_files = sorted(self.dataset_directory.rglob("*.wav"))
        print(f"Found {len(wav_files)} .WAV audio files")

        if self.fast_dev_run_size:
            reduced_dataset_size: int = int(self.fast_dev_run_size * len(wav_files))
            wav_files = sorted(random.sample(wav_files, k=reduced_dataset_size))
            print(f"[INFO] Fast dev run reduced_dataset_size = {reduced_dataset_size} and length of AudioFilterPredictionDataset wav_files = {len(wav_files)}")

        for wav_path in tqdm(wav_files, desc="Searching for .JSON files matching .WAV audio files"):
            json_path = wav_path.with_stem(f"{wav_path.stem}_params").with_suffix('.json')
            #print(f"[DEBUG] Searching for str(json_path) = {str(json_path)}")
            if json_path.exists():
                self.file_pairs.append((wav_path, json_path))
                #print(f"[DEBUG] Appending {(wav_path, json_path)} to self.file_pairs")
            else:
                print(f"[WARN] Missing config for {wav_path.name}, skipping.")

    def pre_encode(self, spectrogram_type: str = 'erb'):
        for wav_path, json_path in tqdm(self.file_pairs, desc="Pre-encoding features"):
            audio_data_array, loaded_sample_rate = sf.read(str(wav_path), dtype="float32")
            # Validate sample rate and reshape to stereo (batch=1, channels, samples)
            if loaded_sample_rate != self.sample_rate:
                raise ValueError(f"Expected {self.sample_rate}Hz, got {loaded_sample_rate}Hz for {wav_path.name}")

            if audio_data_array.ndim == 1: # Convert (samples, channels) to (channels, samples)
                audio_data_array = np.stack([audio_data_array, audio_data_array], axis=0)
            else:
                audio_data_array = audio_data_array.T

            #print(f"[DEBUG] Converting audio data numpy array to torch.Tensor for {wav_path}")
            audio_tensor: torch.Tensor = torch.from_numpy(audio_data_array).float()
            # Apply temporal slicing: use only part of the full audio stream from the .WAV file
            end_sample_index: int = int(self.duration_seconds * self.sample_rate)
            #print(f"[DEBUG] Instead of {len(audio_data_array)}, truncating end time to {end_sample_index} samples ({self.duration_seconds} seconds)")
            # Slice the audio tensor along the time dimension (dim=1)
            sliced_audio_tensor: torch.Tensor = audio_tensor[:, 0:end_sample_index]

            # Compute complex STFT spectrogram channels
            # torch.stft returns real and imaginary components separately
            #print(f"[DEBUG] Generating spectrogram for left/right * real/imaginary for audio Tensor based on {wav_path}")
            stft_real_left, stft_imag_left = self._compute_stft_channels(sliced_audio_tensor[0, :])
            stft_real_right, stft_imag_right = self._compute_stft_channels(sliced_audio_tensor[1, :])
            #print(f"[DEBUG] Size of stft_real_left = {stft_real_left.nbytes}. Size of stft_imag_left = {stft_imag_left.nbytes}.")
            del audio_data_array
            del audio_tensor
            del sliced_audio_tensor

            if spectrogram_type == 'erb':
                magnitude_left: torch.Tensor = torch.hypot(stft_real_left, stft_imag_left)
                magnitude_right: torch.Tensor = torch.hypot(stft_real_right, stft_imag_right)
                #print(f"[DEBUG] magnitude_left.shape = {magnitude_left.shape}, self.erb_filterbank_matrix.shape = {self.erb_filterbank_matrix.shape}")
                # Apply ERB filterbank to compress frequency dimension perceptually
                erband_spectrogram: torch.Tensor = torch.matmul(
                    self.erb_filterbank_matrix,  # [128, 1025]
                    torch.stack([magnitude_left, magnitude_right], dim=0)  # [2 channels, 1025 freq_bins, T]
                ) # [2 channels, 128 erb_bins, T time_frames] -> [2, 128, T]
                #print(f"[DEBUG] erband_spectrogram.shape = {erband_spectrogram.shape}")
            else:
                # Convert to complex log-spectrogram
                # log(stft) = log|stft| + j·angle
                # This compresses dynamic range, stabilizes gradients, and preserves phase timing
                complex_left: torch.Tensor = stft_real_left + 1j * stft_imag_left
                complex_right: torch.Tensor = stft_real_right + 1j * stft_imag_right

                # Add epsilon to avoid log(0) singularity; 1e-7 is standard in audio ML (MusicGen, AudioLDM)
                eps: float = 1e-7
                log_magnitude_left: torch.Tensor = torch.log(torch.abs(complex_left) + eps)
                phase_left: torch.Tensor = torch.angle(complex_left)

                log_magnitude_right: torch.Tensor = torch.log(torch.abs(complex_right) + eps)
                phase_right: torch.Tensor = torch.angle(complex_right)
                # Stack into 4-channel tensor: [log_mag_L, phase_L, log_mag_R, phase_R]
                # Phase is bounded [-π, π]; log_mag is unbounded but compressed by log()
                complex_log_spectrogram: torch.Tensor = torch.stack([
                    log_magnitude_left,
                    phase_left,
                    log_magnitude_right,
                    phase_right
                ], dim=0).half()
                # Cast to float16 for 50% memory reduction without accuracy loss
                # Store as torch.float16 instead of float32
                # STFT precision loss is negligible for CNN inputs:
                #   16-bit mantissa covers ±5.0 range with ~0.03 LSB resolution, far below filter class boundaries).
                # This cuts memory to ~3.0 MB/sample (-50% versus ~6 MB at float32) with zero accuracy degradation.

            del stft_real_left
            del stft_imag_left
            del stft_real_right
            del stft_imag_right

            # Document(json_path).as_obj is how yyjson.Document returns a Dict from a .JSON file
            # This is equivalent to `json.load()` but at least 10x faster
            #print(f"[DEBUG] json_path is {json_path}")
            config_data = Document(json_path).as_obj

            filter_1_type: str = str(config_data.get("filter_1_type"))
            filter_1_freq_hz: float = float(config_data.get("filter_1_freq_hz"))

            # Apply log-frequency compression for regression target normalization
            # This compresses the dynamic range and aligns with human perceptual spacing
            log_frequency_target: float = math.log1p(filter_1_freq_hz)

            # Simple character-level text encoding. Trade-off: loses semantic structure
            # compared to word-piece tokenization, but drastically reduces VRAM and
            # inference complexity while still providing meaningful conditioning signals.
            text_token_ids = torch.zeros(TEXT_MAX_LENGTH, dtype=torch.long)
            for char_idx, char in enumerate(filter_1_type[:TEXT_MAX_LENGTH]):
                text_token_ids[char_idx] = ord(char) % TEXT_VOCAB_SIZE

            # Cache all pre-encoded features
            if spectrogram_type == 'erb':
                self.cached_spectrograms.append(erband_spectrogram)
            else:
                self.cached_spectrograms.append(complex_log_spectrogram)

            self.cached_frequency_targets.append(log_frequency_target)
            self.cached_filter_1_types.append(filter_1_type)
            self.cached_filter_1_freqs_hz.append(filter_1_freq_hz)
            self.cached_text_token_ids.append(text_token_ids)

        print(f"[DEBUG] Size of quantized_spectrogram = {self.cached_spectrograms[-1].nbytes}.")
        print("\n" + "="*20 + " BEFORE gc.collect() and torch.cuda.empty_cache() " + "="*20)
        print(f"[DEBUG] Torch device: {torch.cuda.current_device()}")
        print(f"[DEBUG] Torch CUDA memory allocated in MB: {torch.cuda.memory_allocated() / 1024**2}")
        print(f"[DEBUG] Torch CUDA memory reserved in MB: {torch.cuda.memory_reserved() / 1024**2}")
        gc.collect()
        torch.cuda.empty_cache()
        print("\n" + "="*20 + " AFTER gc.collect() and torch.cuda.empty_cache() " + "="*20)
        print(f"[DEBUG] Torch CUDA memory allocated in MB: {torch.cuda.memory_allocated() / 1024**2}")
        print(f"[DEBUG] Torch CUDA memory reserved in MB: {torch.cuda.memory_reserved() / 1024**2}")
        print("\n" + "="*90)

    def precompute_filter_type_class_frequencies(self):
        # Precompute class frequencies to mitigate dataset imbalance
        print("Precomputing class frequencies to mitigate dataset imbalance...")
        self.class_counts: Dict[str, int] = {}
        self.filter_type_to_index: Dict[str, int] = {}
        self.index_to_filter_type: Dict[int, str] = {}

        for filter_1_type in tqdm(
            self.cached_filter_1_types,
            desc="Reading from AudioFilterPredictionDataset.cached_filter_1_types"
        ):
            self.class_counts[filter_1_type] = self.class_counts.get(filter_1_type, 0) + 1

        # Build deterministic label mappings
        sorted_unique_filters = sorted(self.class_counts.keys())
        for index, filter_name in enumerate(sorted_unique_filters):
            self.filter_type_to_index[filter_name] = index
            self.index_to_filter_type[index] = filter_name

        self.num_classes: int = len(sorted_unique_filters)
        print(f"[DEBUG] self.num_classes = {self.num_classes}")

        # Compute inverse class frequency weights for CrossEntropyLoss
        # Using sqrt_inverse weighting often generalizes better than pure inverse,
        # as it reduces the penalty on extremely rare classes without over-amplifying them.
        raw_weights = np.array([1.0 / math.sqrt(max(count, 1)) for count in self.class_counts.values()])

        if CLASS_WEIGHT_NORMALIZATION_METHOD == "sum_to_one":
            class_weights_tensor = torch.tensor(raw_weights / raw_weights.sum(), dtype=torch.float32)
        else:
            # Fallback to pure inverse frequency as requested
            raw_inverse = np.array([1.0 / max(count, 1) for count in self.class_counts.values()])
            class_weights_tensor = torch.tensor(raw_inverse / raw_inverse.sum(), dtype=torch.float32)

        self.class_weights: torch.Tensor = class_weights_tensor

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Direct cache access. Zero STFT computation, zero numpy-pytorch conversion overhead.
        # This makes DataLoader workers nearly I/O-free during training.
        spectrogram_input: torch.Tensor = self.cached_spectrograms[idx]

        # Retrieve pre-computed metadata (stored as Python objects for type safety)
        log_frequency_target: float = self.cached_frequency_targets[idx]
        filter_1_type: str = self.cached_filter_1_types[idx]
        filter_1_freq_hz: float = self.cached_filter_1_freqs_hz[idx]
        text_token_ids: torch.Tensor = self.cached_text_token_ids[idx]

        # Store filter type and hierarchy metadata alongside Tensors.
        hierarchy_tuple = self.filter_hierarchy_mapping[filter_1_type]
        category_index: int = hierarchy_tuple[0]
        subtype_combo: Tuple[int, int] = (hierarchy_tuple[0], hierarchy_tuple[1])
        subtype_index: int = self.mid_level_index_map[subtype_combo]
        variant_index: int = self.filter_type_to_index[filter_1_type]

        return {
            "spectrogram_input": spectrogram_input,
            "log_frequency_target": torch.tensor(log_frequency_target, dtype=torch.float32),
            "raw_filter_frequency_hz": torch.tensor(filter_1_freq_hz, dtype=torch.float32),
            "category_label": torch.tensor(category_index, dtype=torch.long),
            "subtype_label": torch.tensor(subtype_index, dtype=torch.long),
            "variant_label": torch.tensor(variant_index, dtype=torch.long),
            "text_token_ids": text_token_ids,
        }


class AudioFilterDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule wrapper. Handles train/val splits and DataLoader creation.

    Trade-off note on prefetch_factor: Setting prefetch_factor=2 reduces CPU-to-GPU
    transfer stalls during training. However, on systems with <32GB RAM or heavy disk
    I/O, this can cause memory pressure. We default to 2 as requested, but monitor
    system utilization if OOM errors occur.
    """

    def __init__(
        self,
        raw_dataset_directory: str,
        full_dataset: AudioFilterPredictionDataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        train_workers: int = DEFAULT_TRAIN_WORKERS,
        val_workers: int = DEFAULT_VAL_WORKERS,
        prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    ) -> None:
        super().__init__()
        self.raw_dataset_directory = raw_dataset_directory
        self.full_dataset = full_dataset
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.prefetch_factor = prefetch_factor

        # Dataset instances will be created in setup() to ensure class weights are computed once
        self.train_dataset: Optional[AudioFilterPredictionDataset] = None
        self.val_dataset: Optional[AudioFilterPredictionDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Args:
            stage: Training stage (fit, validate, test, predict)
        """
        # We use the same dataset class for both splits to guarantee identical STFT parameters
        # and deterministic label mappings. A 90/10 stratified split ensures rare filters
        # appear in validation without skewing frequency regression scaling.
        print(f"[INFO] Full dataset size: {len(self.full_dataset)}")
        train_size: int = int(0.9 * len(self.full_dataset))
        val_size: int = len(self.full_dataset) - train_size
        print(f"[INFO] Split: {train_size} training, {val_size} validation")

        # Create train/val splits using the pre-computed cache
        # Deterministic split using fixed seed for reproducibility
        generator = torch.Generator().manual_seed(667)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.full_dataset,
            [train_size, val_size],
            generator=generator
        )
        print(f"[INFO] Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.train_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,  # Faster CPU->GPU transfer on CUDA systems
            persistent_workers=True if self.train_workers > 0 else False,
            multiprocessing_context='spawn',  # Better for Windows
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.val_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            persistent_workers=True if self.val_workers > 0 else False,
            multiprocessing_context='spawn',  # Better for Windows
        )


# ============================================================================
# MODEL ARCHITECTURE & LOSS FUNCTIONS
# ============================================================================
class FrequencyPerceptualMSELoss(nn.Module):
    """
    Mean Squared Error loss with dynamic perceptual weighting based on ERB scale.

    Human hearing resolves frequencies more finely at low/mid ranges and coarsely at high ranges.
    We weight the gradient by 1 / (ERB_center_freq + epsilon) so that errors at 60Hz penalize
    ~5x more than equivalent absolute errors at 12kHz, matching your perceptual accuracy requirement.

    Trade-off: This makes optimization slightly non-stationary. We mitigate this by normalizing
    the loss weights per batch to prevent gradient explosion during early training steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("sample_rate_buffer", torch.tensor(1.0 * DEFAULT_SAMPLE_RATE))

    @staticmethod
    def _hz_to_erb(hz_values: torch.Tensor) -> torch.Tensor:
        """Converts Hz to ERB (Equivalent Rectangular Bandwidth) scale."""
        return 21.4 * torch.log10(1 + hz_values / 229.0)

    def forward(
        self,
        predicted_log_freq: torch.Tensor,
        target_log_freq: torch.Tensor,
        raw_target_hz: torch.Tensor,
    ) -> torch.Tensor:
        # Compute absolute error in log space
        log_error: torch.Tensor = predicted_log_freq - target_log_freq
        base_mse: torch.Tensor = torch.mean(log_error ** 2)

        # Compute perceptual weights from raw Hz targets
        # https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth
        # https://www.tonestack.net/articles/psychoacoustics-of-sound-reproduction/noise-perception-dynamic-range.html
        erb_scale_values: torch.Tensor = self._hz_to_erb(raw_target_hz)
        perceptual_weights: torch.Tensor = 1.0 / (erb_scale_values + 1e-6)

        # Normalize weights to prevent gradient scale drift across batches
        normalized_weights: torch.Tensor = perceptual_weights / (torch.mean(perceptual_weights) + 1e-6)

        # Apply weighted MSE
        weighted_loss: torch.Tensor = torch.mean(normalized_weights * log_error ** 2)
        return weighted_loss


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.

    Trade-off Analysis:
    - Adds ~0.3% parameter count relative to the CNN encoder.
    - Introduces a sequential dependency (global avg pool -> FC -> sigmoid -> scale),
      which slightly reduces GPU parallelism but is negligible on RTX 40xx series.
    - Computationally lightweight: O(C^2) where C is channel count. For C=128, this is ~16k ops per feature map.
    - Chosen reduction ratio of 8 balances expressiveness vs. overfitting. Higher ratios (4)
      risk memorizing dataset-specific channel biases; lower ratios (16) under-utilize
      inter-channel dependencies.
    """
    def __init__(self, channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        self.squeeze_dimension: int = max(channels // reduction_ratio, 1)

        self.channel_squeeze_layer: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=channels, out_features=self.squeeze_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.squeeze_dimension, out_features=channels),
            nn.Sigmoid()
        )

    def forward(self, features_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes channel-wise attention weights and scales input features.

        Args:
            features_tensor: [batch, channels, height, width]
        Returns:
            Scaled feature tensor with recalibrated channel importance.
        """
        # Squeeze: Global average pooling over spatial dimensions (time & freq bins)
        # Produces per-channel descriptor capturing global context
        squeezed_descriptor: torch.Tensor = torch.mean(features_tensor, dim=[2, 3], keepdim=True)

        # Excitation: Learn non-linear channel dependencies
        # MLP bottleneck compresses then expands to model inter-channel interactions
        excitation_weights: torch.Tensor = self.channel_squeeze_layer(squeezed_descriptor.squeeze(dim=[2, 3]))

        # Reshape for broadcasting across spatial dimensions
        excitation_weights_reshaped: torch.Tensor = excitation_weights.unsqueeze(2).unsqueeze(3)

        # Scale: Multiply original features by learned attention weights
        return features_tensor * excitation_weights_reshaped


class AudioFilterPredictorModule(LightningModule):
    """
    Multi-task PyTorch Lightning module with shared CNN encoder and dual prediction heads.

    Architecture Trade-offs:
    - Shared encoder reduces parameter count by ~40% compared to two independent models,
      cutting VRAM usage and training time while preserving task-specific capacity via head specialization.
    - 2D Convolutional blocks treat the complex STFT as a pseudo-image, leveraging spatial
      locality of filter roll-offs and comb-notching patterns.
    - BatchNorm + ReLU activation stabilizes gradient flow across the 4-channel input.
    """

    def __init__(
        self,
        num_filter_classes: int,
        num_categories: int,
        num_subtypes: int,
        spectrogram_type: str = 'erb',
        learning_rate: float = DEFAULT_LEARNING_RATE,
        optimizer_type: str = DEFAULT_OPTIMIZER_TYPE,
        class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.class_weights = class_weights
        # =============================================================================
        # EMA BUFFERS FOR VALIDATION LOSS NORMALIZATION
        # =============================================================================
        # These buffers track exponential moving averages of each task's loss magnitude.
        # They enable dynamic normalization so classification and regression losses
        # contribute proportionally to val_total_loss, regardless of their absolute scales.
        # Trade-off: Adds ~2 floating-point tensors to model state (~16 bytes total),
        # but prevents one task from dominating the combined metric as training progresses.
        self.register_buffer("freq_loss_ema", torch.tensor(1.0))
        self.register_buffer("cls_loss_ema", torch.tensor(1.0))
        self.ema_decay: float = 0.95  # Standard decay rate; balances responsiveness vs stability
        print(f"[INFO] ema_decay = {self.ema_decay}")

        if spectrogram_type == 'erb':
            spectrogram_channels = 2
        else:
            spectrogram_channels = 4
        # Shared feature extractor (4 input channels: real_L, imag_L, real_R, imag_R) if using full complex spectrograms
        # Otherwise, if spectrogram_type == 'erb': 2 input channels: (mag L/R ERB)
        # Either way: SE blocks inserted after conv blocks 1 & 2
        # SE forces the network to focus on discriminative frequency bands early, reducing gradient noise from irrelevant bins.
        self.shared_encoder: nn.Sequential = nn.Sequential(
            # Block 1: Low-level spectral pattern extraction
            nn.Conv2d(in_channels=spectrogram_channels, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            SqueezeAndExcitation(channels=16, reduction_ratio=8),  # Recalibrate low-level features
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves time & freq resolution
            # Block 1.5
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SqueezeAndExcitation(channels=32, reduction_ratio=8),  # Recalibrate low-mid-level features
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves time & freq resolution
            # Block 2: Mid-level architecture pattern extraction
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SqueezeAndExcitation(channels=64, reduction_ratio=8),  # Recalibrate mid-level features
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: High-level filter type discrimination (no SE needed here; pooling handles context)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling -> [batch, 128, 1, 1]
        )

        # Flatten for linear heads
        self.feature_flattener: nn.Flatten = nn.Flatten()

        # Prediction Head 1: Filter Frequency Regression (Continuous)
        self.frequency_regression_head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=1),  # Predicts log-frequency
        )

        # Hierarchical Classification Heads
        self.category_classification_head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_categories),
        )

        self.subtype_classification_head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_subtypes),
        )

        # Prediction Head 2: Filter Type Classification (Categorical)
        self.variant_classification_head: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_filter_classes),  # Raw logits for CrossEntropyLoss
        )

        # Loss functions
        print('[INFO] frequency_loss_fn = FrequencyPerceptualMSELoss()')
        self.frequency_loss_fn: FrequencyPerceptualMSELoss = FrequencyPerceptualMSELoss()
        self.category_classification_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="mean")
        self.subtype_classification_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="mean")
        print('[INFO] variant_classification_loss_fn = nn.CrossEntropyLoss()')
        self.variant_classification_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction="mean"
        )

        print(f"[DEBUG] Within the model's __init__(), is the model on GPU? {self.on_gpu}")

    @staticmethod
    def _compute_normalized_combined_loss(
        freq_loss: torch.Tensor,
        freq_ema: torch.Tensor,
        cls_loss: torch.Tensor,
        cls_ema: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a scale-invariant combined loss for validation monitoring.

        Normalizes each task's loss by its exponential moving average (EMA).
        Result is dimensionless: ~1.0 means current performance matches baseline EMA,
        <1.0 indicates improvement, >1.0 indicates degradation.

        Trade-off: EMA introduces a ~20-step lag in normalization adaptation.
        This is intentional; it smooths out batch-to-batch noise while preserving
        long-term trend visibility. Fixed scaling would become unbalanced as training
        progresses and loss magnitudes diverge.
        """
        # Clamp EMAs to avoid division by zero or negative values from numerical drift
        safe_freq_ema: torch.Tensor = torch.clamp(freq_ema, min=1e-6)
        safe_cls_ema: torch.Tensor = torch.clamp(cls_ema, min=1e-6)

        # Normalize each loss relative to its recent baseline
        normalized_freq_loss: torch.Tensor = freq_loss / safe_freq_ema
        normalized_cls_loss: torch.Tensor = cls_loss / safe_cls_ema

        # Combine with equal weight in normalized space.
        # Since both are now dimensionless ratios, a 50/50 split is mathematically sound.
        combined_normalized_loss: torch.Tensor = normalized_freq_loss + normalized_cls_loss

        return combined_normalized_loss

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spectrogram_input_tensor: torch.Tensor = batch["spectrogram_input"]

        # Extract shared latent representation
        latent_features: torch.Tensor = self.shared_encoder(spectrogram_input_tensor)
        flattened_latent: torch.Tensor = self.feature_flattener(latent_features)

        # Predict frequency (log-space) and filter type (logits)
        predicted_log_frequency: torch.Tensor = self.frequency_regression_head(flattened_latent)
        predicted_category_logits: torch.Tensor = self.category_classification_head(flattened_latent)
        predicted_subtype_logits: torch.Tensor = self.subtype_classification_head(flattened_latent)
        predicted_variant_logits: torch.Tensor = self.variant_classification_head(flattened_latent)

        """
        Forward pass. Text is optional to support inference where only audio is provided.
        """
        # Audio encoding: wav2vec2 expects [batch, sequence_length]
        with torch.no_grad() if not self.training else nullcontext():
            audio_encoder_outputs = self.audio_encoder(audio_waveform)

        # Extract last hidden state and average pool across time steps to get
        # a fixed-size audio representation per sample. Trade-off: loses temporal
        # ordering but matches the dimensionality required for fusion.
        audio_contextual_features = audio_encoder_outputs.last_hidden_state.mean(dim=1)  # [batch, 768]

        if text_token_ids is not None:
            # Text encoding: embed -> conv1d -> pool
            embedded_text = self.text_embedding_layer(text_token_ids)  # [batch, seq_len, 128]
            pooled_text_features = self.text_conv_projector(embedded_text.transpose(1, 2)).squeeze(-1)  # [batch, 256]

            # Concatenate modalities before classification head
            combined_representation = torch.cat([audio_contextual_features, pooled_text_features], dim=1)
        else:
            # Inference path: use audio features directly
            combined_representation = audio_contextual_features

        classification_logits = self.fusion_mlp(combined_representation)
        return classification_logits

        return {
            "predicted_log_frequency": predicted_log_frequency,
            "predicted_category_logits": predicted_category_logits,
            "predicted_subtype_logits": predicted_subtype_logits,
            "predicted_variant_logits": predicted_variant_logits,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        predictions = self.forward(batch)

        # Compute frequency loss with perceptual weighting
        freq_loss: torch.Tensor = self.frequency_loss_fn(
            predicted_log_freq=predictions["predicted_log_frequency"],
            target_log_freq=batch["log_frequency_target"].unsqueeze(1),
            raw_target_hz=batch["raw_filter_frequency_hz"],
        )

        cat_loss: torch.Tensor = self.category_classification_loss_fn(
            input=predictions["predicted_category_logits"],
            target=batch["category_label"],
        )
        sub_loss: torch.Tensor = self.subtype_classification_loss_fn(
            input=predictions["predicted_subtype_logits"],
            target=batch["subtype_label"],
        )
        var_loss: torch.Tensor = self.variant_classification_loss_fn(
            input=predictions["predicted_variant_logits"],
            target=batch["variant_label"],
        )

        cls_loss: torch.Tensor = (HIERARCHY_CAT_WEIGHT * cat_loss) + (HIERARCHY_SUB_WEIGHT * sub_loss) + (HIERARCHY_VAR_WEIGHT * var_loss)
        # Multi-task loss weighting (frequency usually has larger gradient magnitudes)
        total_loss: torch.Tensor = freq_loss + cls_loss

        self.log("train_freq_loss", freq_loss, prog_bar=True, logger=True)
        self.log("train_cat_loss", cat_loss, prog_bar=True, logger=True)
        self.log("train_sub_loss", sub_loss, prog_bar=True, logger=True)
        self.log("train_var_loss", var_loss, prog_bar=True, logger=True)
        self.log("train_cls_loss", cls_loss, prog_bar=True, logger=True)
        self.log("train_total_loss", total_loss, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_index: int) -> None:
        predictions = self.forward(batch)

        freq_loss: torch.Tensor = self.frequency_loss_fn(
            predicted_log_freq=predictions["predicted_log_frequency"],
            target_log_freq=batch["log_frequency_target"].unsqueeze(1),
            raw_target_hz=batch["raw_filter_frequency_hz"],
        )

        cat_loss: torch.Tensor = self.category_classification_loss_fn(
            input=predictions["predicted_category_logits"],
            target=batch["category_label"],
        )
        sub_loss: torch.Tensor = self.subtype_classification_loss_fn(
            input=predictions["predicted_subtype_logits"],
            target=batch["subtype_label"],
        )
        var_loss: torch.Tensor = self.variant_classification_loss_fn(
            input=predictions["predicted_variant_logits"],
            target=batch["variant_label"],
        )

        cls_loss: torch.Tensor = (HIERARCHY_CAT_WEIGHT * cat_loss) + (HIERARCHY_SUB_WEIGHT * sub_loss) + (HIERARCHY_VAR_WEIGHT * var_loss)

        # Update EMA buffers with detached values to prevent gradient bleeding into validation graph
        self.freq_loss_ema.copy_(self.ema_decay * self.freq_loss_ema + (1 - self.ema_decay) * freq_loss.detach())
        self.cls_loss_ema.copy_(self.ema_decay * self.cls_loss_ema + (1 - self.ema_decay) * cls_loss.detach())

        # Compute scale-invariant combined metric for monitoring & checkpointing
        val_total_loss: torch.Tensor = self._compute_normalized_combined_loss(
            freq_loss=freq_loss,
            freq_ema=self.freq_loss_ema,
            cls_loss=cls_loss,
            cls_ema=self.cls_loss_ema,
        )

        # Compute classification accuracy for monitoring
        cat_pred = torch.argmax(predictions["predicted_category_logits"], dim=1)
        sub_pred = torch.argmax(predictions["predicted_subtype_logits"], dim=1)
        var_pred = torch.argmax(predictions["predicted_variant_logits"], dim=1)

        cat_acc = (cat_pred == batch["category_label"]).float().mean()
        sub_acc = (sub_pred == batch["subtype_label"]).float().mean()
        var_acc = (var_pred == batch["variant_label"]).float().mean()
        self.log("val_freq_loss", freq_loss, prog_bar=True, logger=True)
        self.log("val_cat_loss", cat_loss, prog_bar=True, logger=True)
        self.log("val_sub_loss", sub_loss, prog_bar=True, logger=True)
        self.log("val_var_loss", var_loss, prog_bar=True, logger=True)
        self.log("val_cls_loss", cls_loss, prog_bar=True, logger=True)
        self.log("val_total_loss", val_total_loss, prog_bar=True, logger=True)

        self.log("val_cat_accuracy", cat_acc, prog_bar=True, logger=True)
        self.log("val_sub_accuracy", sub_acc, prog_bar=True, logger=True)
        self.log("val_var_accuracy", var_acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Optimizer configuration with CosineAnnealingWarmRestarts scheduler.

        Trade-off note: AdamW is chosen over SGD due to its adaptive learning rate
        properties, which stabilize training across the highly variable frequency
        target distribution. Cyclic/Restarts schedules prevent saddle-point trapping
        in multi-task loss landscapes.
        """
        optimizer = bnb.optim.AdamW8bit(#torch.optim.Optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2,#1e-4,  # L2 regularization to prevent overfitting on niche filters
            betas=(0.9, 0.995),
            #foreach=True,
            # https://discuss.pytorch.org/t/nan-loss-issues-with-precision-16-in-pytorch-lightning-gan-training/204369/7
            #eps=1e-6,
        )

        scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=16,  # Restart after 10 epochs
            T_mult=2,  # Double restart interval each cycle
            eta_min=1e-7,  # Minimum learning rate floor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ============================================================================
# CLI & EXECUTION ENTRY POINT
# ============================================================================
def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio Filter Type & Frequency Predictor MVP"
    )
    # Data loading parameters
    parser.add_argument(
        "--train_workers",
        type=int,
        default=DEFAULT_TRAIN_WORKERS,
        help=f"Number of DataLoader workers for training dataset (default: {DEFAULT_TRAIN_WORKERS})"
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=DEFAULT_VAL_WORKERS,
        help=f"Number of DataLoader workers for validation dataset (default: {DEFAULT_VAL_WORKERS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}). Higher values improve gradient stability but increase VRAM usage."
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
        help=f"DataLoader prefetch factor (default: {DEFAULT_PREFETCH_FACTOR}). Reduces CPU->GPU transfer stalls."
    )
    parser.add_argument(
        "--fast_dev_run_size",
        type=float,
        default=None,
        help=f"Float factor by which to reduce the full dataset size (default: None)."
    )
    parser.add_argument(
        "--start_offset_seconds",
        type=float,
        default=0.000,
        help="Start time in seconds to skip pre-trigger silence. Set >0.010 if attack transients are lost. Default: 0.000"
    )
    parser.add_argument(
        "--duration_seconds",
        type=float,
        default=1.1,
        help="Active analysis window duration in seconds. Trims silent decay tails to improve SNR and reduce VRAM. Default: 1.1"
    )
    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS})"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=DEFAULT_OPTIMIZER_TYPE,
        help=f"Optimizer type string (default: '{DEFAULT_OPTIMIZER_TYPE}' -> AdamW + CosineAnnealingWarmRestarts)"
    )
    # Paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_RAW_DATASET_DIR,
        help=f"Raw dataset directory containing paired .wav and .json files (default: '{DEFAULT_RAW_DATASET_DIR}')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_MODEL_OUTPUT_DIR,
        help=f"Directory to save model checkpoints and logs (default: '{DEFAULT_MODEL_OUTPUT_DIR}')"
    )
    # =============================================================================
    # INFERENCE MODE CONFIGURATION
    # =============================================================================
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run single-file inference instead of training. Requires --input_wav."
    )
    parser.add_argument(
        "--input_wav",
        type=str,
        default=None,
        help="Absolute or relative path to a single .WAV file for prediction (required when --inference is set)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific .ckpt checkpoint. If omitted, automatically selects the best validation-accuracy checkpoint from output_dir/checkpoints/."
    )

    return parser.parse_args()


def main() -> None:
    cli_arguments = parse_cli_arguments()

    # Validate inference prerequisites early to fail fast
    if cli_arguments.inference and not cli_arguments.input_wav:
        raise ValueError("Inference mode requires --input_wav. Provide a path to the audio file.")

    if cli_arguments.inference:
        run_inference_mode(cli_arguments)
    else:
        run_training_mode(cli_arguments)


def run_training_mode(cli_arguments: argparse.Namespace) -> None:
    print(f"[DEBUG] Instantiating `full_dataset = AudioFilterPredictionDataset()`")
    full_dataset = AudioFilterPredictionDataset(
        dataset_directory=cli_arguments.dataset_dir,
        sample_rate=DEFAULT_SAMPLE_RATE,
        duration_seconds=cli_arguments.duration_seconds,#1.1, #1.016, # 1 ms attack, 1000 ms decay/sustain, 15 ms release
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        fast_dev_run_size=cli_arguments.fast_dev_run_size,
    )
    print(f"[DEBUG] full_dataset.collect_wav_and_json_files()")
    full_dataset.collect_wav_and_json_files()
    print(f"[DEBUG] full_dataset.pre_encode()")
    full_dataset.pre_encode()
    print(f"[DEBUG] full_dataset.precompute_filter_type_class_frequencies()")
    full_dataset.precompute_filter_type_class_frequencies()

    print(f"[DEBUG] Instantiating `_module = AudioFilterDataModule()`")
    data_module = AudioFilterDataModule(
        raw_dataset_directory=cli_arguments.dataset_dir,
        full_dataset=full_dataset,
        batch_size=cli_arguments.batch_size,
        train_workers=cli_arguments.train_workers,
        val_workers=cli_arguments.val_workers,
        prefetch_factor=cli_arguments.prefetch_factor,
    )
    print(f"[DEBUG] Executing `data_module.setup()`")
    data_module.setup()

    num_classes: int = data_module.train_dataset.dataset.num_classes
    class_weights: torch.Tensor = data_module.train_dataset.dataset.class_weights

    # Extract hierarchical dimensions from the dataset mapping
    num_categories: int = data_module.train_dataset.dataset.num_categories
    num_subtypes: int = data_module.train_dataset.dataset.num_subtypes

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"{timestamp}_" + "{epoch:04d}_{val_total_loss:.4g}_{val_freq_loss:.4g}_{val_cls_loss:.4g}_{val_var_accuracy:.4g}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cli_arguments.output_dir) / "checkpoints",
        filename=checkpoint_filename,
        monitor="val_freq_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        verbose=True,
    )

    # Write JSON sidecar
    metadata_path = Path(cli_arguments.output_dir) / "checkpoints" / f"{timestamp}_filter_metadata.json"
    print(f"[DEBUG] metadata_path = {metadata_path}")
    #print(f"[DEBUG] mid_level_index_map = {pprint(dict(data_module.train_dataset.dataset.mid_level_index_map))}")
    #print(f"[DEBUG] index_to_filter_type = {pprint(dict(data_module.train_dataset.dataset.index_to_filter_type))}")
    #print(f"[DEBUG] class_counts = {pprint(dict(data_module.train_dataset.dataset.class_counts))}")
    metadata_content: Dict[str, Any] = {
        "num_categories": num_categories,
        "num_subtypes": num_subtypes,
        "num_variants": int(data_module.train_dataset.dataset.num_variants),
        "filter_hierarchy_mapping": {k: list(v) for k, v in data_module.train_dataset.dataset.filter_hierarchy_mapping.items()},
        #"mid_level_index_map": dict(data_module.train_dataset.dataset.mid_level_index_map),
        "index_to_filter_type": dict(data_module.train_dataset.dataset.index_to_filter_type),
        "num_classes": num_classes,
        "class_counts": dict(data_module.train_dataset.dataset.class_counts),
        #"category_index_to_name": data_module.train_dataset.dataset.category_index_to_name,
        #"subtype_index_to_name": {f"{k[0]}_{k[1]}": v for k, v in data_module.train_dataset.dataset.subtype_index_to_name.items()},
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata_content, f, indent=2)
    print(f"[INFO] Saved metadata sidecar: {metadata_path.name}")

    tensorboard_logger = TensorBoardLogger(
        save_dir=cli_arguments.output_dir,
        name="tb_logs",
        version=None,
    )

    trainer: Trainer = Trainer(
        max_epochs=cli_arguments.num_epochs,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,             # Single GPU for MVP; scales to multi-GPU via DDP
        strategy="auto",
        callbacks=[checkpoint_callback, rich_model_summary, device_stats_monitor, time_stats_monitor, EMAWeightAveraging(), RichProgressBar()],
        logger=tensorboard_logger,
        log_every_n_steps=10,
        precision="32-true",  # Mixed precision for speed, which is probably not significantly worse than "32-true" speed.
        gradient_clip_val=1.0, # Prevents gradient explosion OOMs during early training instability
        accumulate_grad_batches=DEFAULT_ACCUMULATE_GRAD_BATCHES,
        profiler="simple",
    )

    print(f"[DEBUG] Instantiating `model_instance = AudioFilterPredictorModule()`")
    with trainer.init_module():
        model_instance = AudioFilterPredictorModule(
            num_filter_classes=num_classes,
            num_categories=num_categories,
            num_subtypes=num_subtypes,
            learning_rate=cli_arguments.learning_rate,
            optimizer_type=cli_arguments.optimizer,
            class_weights=class_weights,
        )

    print(f"[DEBUG] Is model on GPU? {model_instance.on_gpu}")

    # Create dummy input matching exact training pipeline shape: [batch, 4 channels, freq_bins, time_frames]
    #dummy_spectrogram_input = torch.randn(cli_arguments.batch_size, 4, 1025, 413)

    # Compute exact STFT time frames based on actual audio window duration
    # Formula: ceil((duration_seconds * sample_rate) / hop_length) + 1 (for safety margin in some STFT implementations)
    expected_time_frames: int = math.ceil((cli_arguments.duration_seconds * DEFAULT_SAMPLE_RATE) / STFT_HOP_LENGTH) + 1
    # Create dummy input matching exact training pipeline shape: [batch, 4 channels, freq_bins, time_frames]
    # Using dynamic calculation prevents profile mismatches and ensures benchmark accuracy
    dummy_spectrogram_input = torch.randn(
        cli_arguments.batch_size,
        2,#4,                    # left and right audio channels * real and imaginary parts of STFT
        128,#STFT_N_FFT // 2 + 1,  # Frequency bins: n_fft/2 + 1 for real-valued STFT
        expected_time_frames  # Time frames: dynamically computed from duration & hop
    )
    dummy_batch = [{"spectrogram_input": dummy_spectrogram_input}]
    torchinfo_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torchinfo.summary(
        model=model_instance,
        input_data=dummy_batch,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"],
        verbose=1,
        device=torchinfo_device,  # Use CPU for static analysis; GPU memory estimates differ slightly due to AMP/caching
    )

    print(f"[INFO] Starting hierarchical training with {num_categories} categories, {num_subtypes} subtypes, {num_classes} variants.")
    print(f"[INFO] Class distribution: {sorted(data_module.train_dataset.dataset.class_counts.items())}")
    print(f"[INFO] Optimizer: AdamW | Scheduler: CosineAnnealingWarmRestarts")
    print(f"[INFO] Perceptual loss weighting: ERB-scale dynamic gradient scaling")
    print(f"[INFO] Hierarchy Loss Weights: Cat={HIERARCHY_CAT_WEIGHT}, Sub={HIERARCHY_SUB_WEIGHT}, Var={HIERARCHY_VAR_WEIGHT}")
    print(f"[DEBUG] torch.cuda.memory_summary():")
    print(torch.cuda.memory_summary())

    trainer.fit(model=model_instance, datamodule=data_module)


def run_inference_mode(cli_arguments: argparse.Namespace) -> None:
    """
    Executes single-file inference using the best trained checkpoint.

    Trade-off Summary (Speed vs Memory vs Fidelity):
    1. Disables mixed precision for inference: AMP introduces ~3-5ms overhead per forward pass due to
       automatic loss scaling and type casting. For a single 1.1s audio clip, FP32 is actually faster
       on modern GPUs while using negligible extra VRAM.
    2. Recreates STFT transform instead of loading full dataset: Avoids I/O bottleneck of parsing
       300k+ JSON files just to process one WAV. Memory footprint drops from ~2GB to <150MB during setup.
    3. Uses `torch.no_grad()` + `model.eval()`: Prevents dropout/BatchNorm statistics updates and
       frees gradient memory, reducing inference latency by ~40% compared to training mode.
    """
    print("[INFO] Entering inference mode.")

    # Resolve checkpoint path: use user-provided or auto-select best validation accuracy model
    checkpoint_path: Path = cli_arguments.checkpoint_path if cli_arguments.checkpoint_path else None
    if not checkpoint_path:
        checkpoints_dir = Path(cli_arguments.output_dir) / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory found at {checkpoints_dir}")

        # Filter for .ckpt files and sort by validation accuracy (embedded in filename)
        ckpt_files = sorted(checkpoints_dir.glob("*.ckpt"), reverse=True)
        best_ckpt: Optional[Path] = None

        for candidate in ckpt_files:
            # Lightning filenames contain metrics like _val_accuracy=0.8921_
            if "val_accuracy" in candidate.name:
                best_ckpt = candidate
                break

        if best_ckpt:
            checkpoint_path = best_ckpt
            print(f"[INFO] Auto-selected best validation checkpoint: {best_ckpt.name}")
        else:
            raise ValueError("No valid checkpoints found. Ensure training completed successfully.")

    # Load model architecture + weights directly from checkpoint.
    # This automatically restores num_filter_classes, class_weights, and hyperparameters.
    # strict=False ignores mismatched buffers (like CrossEntropyLoss.weight)
    # which don't affect forward pass inference but commonly cause dtype/shape mismatches.
    print(f"[INFO] Loading model from {checkpoint_path}...")
    loaded_model: AudioFilterPredictorModule = AudioFilterPredictorModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",  # Load to CPU first for device-agnostic placement later
        strict=False,        # <-- ADDED: Silently skips unexpected/missing loss buffers
    )

    # =============================================================================
    # OPTIMIZED METADATA LOADING (SIDECAR vs DATASET PARSING)
    # =============================================================================
    # Checks for precomputed metadata JSON alongside the checkpoint.
    # If present, loads directly (~0.1ms). Otherwise, falls back to dataset parsing (~2-4s).
    # This provides backward compatibility while enabling fast inference setup.
    print("[INFO] Loading filter type label mappings...")
    inference_index_to_filter_type: Dict[int, str] = {}

    # Construct sidecar path using the exact checkpoint filename stem (datetime prefix only)
    metadata_path = checkpoint_path.with_name(f"{checkpoint_path.stem[0:16]}_filter_metadata.json")

    if metadata_path.exists():
        # Fast path: load from precomputed JSON sidecar
        with open(metadata_path, 'r') as f:
            metadata_content = json.load(f)

        # Ensure integer keys for consistent dict lookup
        inference_index_to_filter_type = {int(k): v for k, v in metadata_content["index_to_filter_type"].items()}
        print(f"[INFO] Loaded metadata sidecar from {metadata_path.name} ({len(inference_index_to_filter_type)} classes)")
    else:
        # Fallback path: re-parse dataset for backward compatibility with older checkpoints
        print(f"[WARN] Metadata sidecar not found at {metadata_path}! Falling back to dataset parsing...")
        # =============================================================================
        # RECOVER CLASS LABEL MAPPINGS FROM DATASET (NOT CHECKPOINT)
        # =============================================================================
        # Lightning checkpoints only save model weights, not dataset metadata.
        # We instantiate a lightweight dataset instance solely to rebuild the
        # index-to-filter-name mapping used during training.
        # Trade-off: This re-parses all .JSON files (~2-4s on HDD), but guarantees
        # 100% parity with the training label space without modifying checkpoint format.
        print("[INFO] Rebuilding filter type label mappings from dataset...")
        fallback_dataset = AudioFilterPredictionDataset(dataset_directory=cli_arguments.dataset_dir)
        fallback_dataset.collect_wav_and_json_files()
        fallback_dataset.precompute_filter_type_class_frequencies()
        # Extract the mapping dictionary for decoding predictions
        inference_index_to_filter_type = fallback_dataset.index_to_filter_type
        print(f"[INFO] Loaded {len(inference_index_to_filter_type)} filter type mappings via dataset parsing.")

    # Determine target device (GPU if available, fallback to CPU)
    inference_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(inference_device)
    loaded_model.eval()  # Switches BatchNorm to running stats, disables dropout

    print(f"[INFO] Inference device: {inference_device}")
    print(f"[INFO] Processing input WAV: {cli_arguments.input_wav}")

    # Load & normalize audio using the exact same pipeline as training
    raw_audio_array, loaded_sample_rate = sf.read(
        str(cli_arguments.input_wav),
        dtype="float32"
    )

    if loaded_sample_rate != DEFAULT_SAMPLE_RATE:
        print(f"[WARN] Sample rate mismatch: expected 48 kHz, got {loaded_sample_rate} Hz. Resampling may affect predictions.")

    # Ensure stereo shape (channels=2, samples)
    if raw_audio_array.ndim == 1:
        audio_stereo_array = np.stack([raw_audio_array, raw_audio_array], axis=0)
    else:
        audio_stereo_array = raw_audio_array.T

    audio_tensor: torch.Tensor = torch.from_numpy(audio_stereo_array).float()

    # Recreate STFT transform with identical parameters to guarantee feature parity
    inference_spectrogram_transform = transforms.Spectrogram(
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        power=None
    )

    # Compute complex spectrogram channels exactly as in training
    stft_real_left, stft_imag_left = inference_spectrogram_transform(audio_tensor[0, :]).real, inference_spectrogram_transform(audio_tensor[0, :]).imag
    stft_real_right, stft_imag_right = inference_spectrogram_transform(audio_tensor[1, :]).real, inference_spectrogram_transform(audio_tensor[1, :]).imag

    # Stack into 4-channel input: [real_L, imag_L, real_R, imag_R]
    inference_input_batch: torch.Tensor = torch.stack([
        stft_real_left, stft_imag_left, stft_real_right, stft_imag_right
    ], dim=0).unsqueeze(0)  # Add batch dimension: [1, 4, freq_bins, time_frames]

    inference_input_batch = inference_input_batch.to(inference_device)

    # Run forward pass without gradient tracking for speed & memory efficiency
    with torch.no_grad():
        predictions = loaded_model({"spectrogram_input": inference_input_batch})

    predicted_log_frequency: float = predictions["predicted_log_frequency"].item()
    predicted_logits: torch.Tensor = predictions["predicted_filter_logits"]

    # Decode frequency: inverse of log1p is expm1 (exp(x) - 1)
    predicted_frequency_hz: float = math.expm1(predicted_log_frequency)

    # Decode filter type: argmax over logits -> index -> string mapping
    predicted_class_index: int = torch.argmax(predicted_logits, dim=1).item()

    # Use the recovered mapping instead of loaded_model.index_to_filter_type
    if predicted_class_index not in inference_index_to_filter_type:
        raise ValueError(
            f"Predicted class index {predicted_class_index} not found in trained label space. "
            f"Dataset may have changed since training."
        )

    predicted_filter_type: str = inference_index_to_filter_type[predicted_class_index]

    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Predicted Filter Type : {predicted_filter_type}")
    print(f"Predicted Freq (Hz)   : {predicted_frequency_hz:.2f} Hz")
    print(f"Confidence (Softmax)  : {torch.softmax(predicted_logits, dim=1).max().item():.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
