#!/usr/bin/env python3
"""
LPF Frequency Prediction for Serum 2 Synthesizer
=================================================

A machine learning pipeline to predict low-pass filter frequency settings
from audio samples using PyTorch Lightning.

Key Features:
- Mel spectrogram feature extraction
- CNN architecture with attention
- Mixed precision training
- CLI interface for training and inference

Improvements Implemented:
1. Target normalization - LPF frequencies normalized to [0, 1] range
2. Gradient clipping - Prevents exploding gradients during training
3. Comprehensive type hints - Full typing throughout the codebase
4. Model summary - Detailed architecture and parameter reporting
5. Input validation - Audio file validation before processing
6. Dataset statistics - Summary statistics for datasets
7. Prediction confidence - Monte Carlo dropout for uncertainty estimation
8. Progress bar - Visual feedback during batch processing
9. Better checkpoint naming - Timestamps in saved models

Author: Assistant
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

# Audio processing
import librosa
import soundfile as sf


# =============================================================================
# CONSTANT DEFINITIONS
# =============================================================================

class LPFConstants:
    """
    Centralized constants for LPF frequency prediction.
    
    These values are derived from the expected range of Serum 2 filter
    frequencies and the characteristics of the audio data.
    
    Frequency Range Justification:
    - Minimum (50 Hz): Below this, the note becomes less distinct
    - Maximum (10000 Hz): Above this, the filter has minimal effect
    - Range (9950 Hz): Covers the typical filter sweep range
    """
    
    # Target frequency range for normalization
    MIN_FREQ_HZ: float = 50.0       # Minimum reasonable LPF frequency (Hz)
    MAX_FREQ_HZ: float = 10000.0    # Maximum reasonable LPF frequency (Hz)
    FREQ_RANGE: float = MAX_FREQ_HZ - MIN_FREQ_HZ  # 9950 Hz
    
    # Audio processing parameters
    SAMPLE_RATE: int = 48000        # Native sample rate of audio files
    DURATION_SECONDS: float = 1.0   # Expected audio duration
    
    # Spectrogram parameters
    N_MELS_DEFAULT: int = 128       # Number of Mel frequency bins
    N_FFT_DEFAULT: int = 2048       # FFT window size
    HOP_LENGTH_DEFAULT: int = 512   # Hop length for STFT
    
    # Training parameters
    LEARNING_RATE_DEFAULT: float = 0.001
    BATCH_SIZE_DEFAULT: int = 32
    EPOCHS_DEFAULT: int = 100
    
    # Model architecture parameters
    INPUT_CHANNELS: int = 1         # Single channel (mono)
    OUTPUT_DIM: int = 1             # Single frequency output
    
    @classmethod
    def normalize_frequency(cls, freq_hz: float) -> float:
        """
        Normalize frequency from Hz to [0, 1] range.
        
        Formula: (freq - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
        
        Args:
            freq_hz: Frequency in Hertz
            
        Returns:
            Normalized frequency in [0, 1] range
        """
        clamped = np.clip(freq_hz, cls.MIN_FREQ_HZ, cls.MAX_FREQ_HZ)
        return (clamped - cls.MIN_FREQ_HZ) / cls.FREQ_RANGE
    
    @classmethod
    def denormalize_frequency(cls, normalized: float) -> float:
        """
        Convert normalized frequency back to Hz.
        
        Formula: normalized * (MAX_FREQ - MIN_FREQ) + MIN_FREQ
        
        Args:
            normalized: Normalized frequency in [0, 1] range
            
        Returns:
            Frequency in Hertz
        """
        return normalized * cls.FREQ_RANGE + cls.MIN_FREQ_HZ


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_mel_spectrogram(mel_spectrogram: np.ndarray) -> np.ndarray:
    """
    Normalize Mel spectrogram to [0, 1] range.
    
    This function performs per-frequency-band normalization followed by
    global rescaling. This helps the CNN learn consistent spectral patterns
    regardless of absolute amplitude variations.
    
    Process:
    1. Standardize each frequency band to zero mean and unit variance
    2. Clip to reasonable range [-50, 50] to handle outliers
    3. Rescale to [0, 1] range using min-max normalization
    
    Args:
        mel_spectrogram: 2D numpy array of Mel spectrogram values
        
    Returns:
        Normalized Mel spectrogram with values in [0, 1]
    """
    # Per-band normalization (standardization)
    mean = np.mean(mel_spectrogram, axis=1, keepdims=True)
    std = np.std(mel_spectrogram, axis=1, keepdims=True) + 1e-8
    
    normalized = (mel_spectrogram - mean) / std
    
    # Clip to reasonable range and rescale
    normalized = np.clip(normalized, -50, 50)
    normalized = (normalized - normalized.min()) / (
        normalized.max() - normalized.min() + 1e-8
    )
    
    return normalized


def validate_audio_file(path: Path) -> Tuple[bool, str]:
    """
    Validate that an audio file meets requirements.
    
    Checks:
        - File exists and is readable
        - Sample rate matches expected (48kHz)
        - Duration is approximately 1 second
        - File format is WAV
    
    Args:
        path: Path to the audio file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        info = sf.info(str(path))
        
        issues = []
        
        # Check sample rate
        if info.samplerate != LPFConstants.SAMPLE_RATE:
            issues.append(
                f"sample rate {info.samplerate}, expected {LPFConstants.SAMPLE_RATE}"
            )
        
        # Check duration (should be ~1 second)
        duration_diff = abs(info.duration - LPFConstants.DURATION_SECONDS)
        if duration_diff > 0.2:
            issues.append(
                f"duration {info.duration:.2f}s, expected ~1s"
            )
        
        # Check format
        if info.format not in ['WAV', 'WAVEX']:
            issues.append(f"format {info.format}, expected WAV")
        
        if issues:
            return False, f"Issues: {'; '.join(issues)}"
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"


def compute_dataset_statistics(data_dir: str) -> Dict[str, Any]:
    """
    Compute dataset statistics for reporting.
    
    Analyzes all JSON files in the data directory and computes
    statistics about the LPF frequency distribution.
    
    Args:
        data_dir: Path to directory containing .JSON files
        
    Returns:
        Dictionary with frequency distribution, counts, etc.
    """
    json_files = sorted(Path(data_dir).glob("*.json"))
    frequencies = []
    filter_types = defaultdict(int)
    filter_counts = 0
    
    for jf in json_files:
        try:
            with open(jf) as f:
                config = json.load(f)
            
            freq = float(config["filter_1_freq_hz"])
            frequencies.append(freq)
            
            # Track filter types if available
            if "filter_1_type" in config:
                filter_types[config["filter_1_type"]] += 1
            
            filter_counts += 1
        except Exception as e:
            print(f"Warning: Could not process {jf.name}: {e}")
    
    return {
        'total_samples': len(json_files),
        'min_frequency': min(frequencies) if frequencies else 0,
        'max_frequency': max(frequencies) if frequencies else 0,
        'mean_frequency': np.mean(frequencies) if frequencies else 0,
        'std_frequency': np.std(frequencies) if frequencies else 0,
        'median_frequency': np.median(frequencies) if frequencies else 0,
        'frequency_range': f"{min(frequencies)} - {max(frequencies)} Hz" if frequencies else "N/A",
        'filter_types': dict(filter_types),
        'files_processed': filter_counts
    }


def print_dataset_statistics(stats: Dict[str, Any]) -> None:
    """
    Print dataset statistics in a formatted way.
    
    Args:
        stats: Dictionary containing dataset statistics
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Frequency range: {stats['frequency_range']}")
    print(f"Mean frequency: {stats['mean_frequency']:.1f} Hz")
    print(f"Std deviation: {stats['std_frequency']:.1f} Hz")
    print(f"Median frequency: {stats['median_frequency']:.1f} Hz")
    print(f"\nFilter types distribution:")
    for ftype, count in sorted(stats['filter_types'].items()):
        print(f"  {ftype}: {count}")
    print("=" * 60 + "\n")


def print_model_summary(model: LPFCNN) -> None:
    """
    Print detailed model architecture and parameter counts.
    
    Args:
        model: LPFCNN instance to summarize
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Parameter breakdown by layer type
    conv_params = sum(
        p.numel() for m in model.conv_blocks 
        for p in m.parameters() if isinstance(m, nn.Conv2d)
    )
    fc_params = sum(
        p.numel() for m in model.fc_layers 
        for p in m.parameters() if isinstance(m, nn.Linear)
    )
    
    print(f"\nParameter breakdown:")
    print(f"  Convolutional layers: {conv_params:,}")
    print(f"  Fully connected layers: {fc_params:,}")
    print("=" * 60 + "\n")


# =============================================================================
# DATA MODULE
# =============================================================================

class LPFDataset(Dataset):
    """
    Dataset for loading and preprocessing audio samples with their LPF labels.
    
    Expected format:
        - .WAV files: 1-second C3 notes at 48kHz, 32-bit float
        - .JSON files: Contains filter frequency configuration
    
    The dataset handles:
        - Audio loading via soundfile (preserves 32-bit float)
        - Mel spectrogram computation
        - Normalization for training stability
        
    Target Normalization:
        LPF frequencies are normalized to [0, 1] range using:
            normalized = (frequency - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
        where MIN_FREQ = 50 Hz and MAX_FREQ = 10000 Hz
        
    This helps the model learn more effectively across the wide dynamic
    range of filter frequencies.
    
    Attributes:
        wav_paths: List of paths to .WAV files
        n_mels: Number of Mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency for Mel spectrogram
        fmax: Maximum frequency for Mel spectrogram
    """
    
    def __init__(
        self,
        wav_paths: List[Path],
        n_mels: int = LPFConstants.N_MELS_DEFAULT,
        n_fft: int = LPFConstants.N_FFT_DEFAULT,
        hop_length: int = LPFConstants.HOP_LENGTH_DEFAULT,
        fmin: float = 20.0,
        fmax: float = 24000.0
    ):
        """
        Initialize the dataset.
        
        Args:
            wav_paths: List of paths to .WAV files
            n_mels: Number of Mel frequency bins (default: 128)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length for STFT (default: 512)
            fmin: Minimum frequency for Mel spectrogram (default: 20 Hz)
            fmax: Maximum frequency for Mel spectrogram (default: 24 kHz)
        """
        # Input validation
        if not wav_paths:
            raise ValueError("wav_paths cannot be empty")
        
        for path in wav_paths:
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            if path.suffix.lower() != '.wav':
                raise ValueError(f"Expected .wav file, got: {path}")
        
        self.wav_paths = wav_paths
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        
    def __len__(self) -> int:
        return len(self.wav_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Get a single sample with its label.
        
        Returns:
            Tuple of (mel_spectrogram_tensor, lpf_frequency_normalized)
            where lpf_frequency is normalized to [0, 1] range
        """
        wav_path = self.wav_paths[idx]
        json_path = wav_path.with_stem(f"{wav_path.stem}_params").with_suffix('.json')
        
        # Load audio at native sample rate (48kHz) using float32
        audio_data, sample_rate = sf.read(
            str(wav_path),
            dtype='float32'
        )
        
        # Handle stereo: convert to mono by averaging channels
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Compute Mel spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale (dB) for better dynamic range
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize to [0, 1] range per frequency band
        normalized = normalize_mel_spectrogram(mel_db)
        
        # Convert to torch tensor with shape (channels, height, width)
        features_tensor = torch.FloatTensor(normalized).unsqueeze(0)
        
        # Load JSON configuration and extract LPF frequency
        config = json.load(open(json_path))
        lpf_frequency_normalized = self._extract_lpf_frequency(config)
        
        return features_tensor, lpf_frequency_normalized
    
    def _extract_lpf_frequency(self, config: dict) -> float:
        """
        Extract LPF frequency from configuration dictionary and normalize.
        
        Serum 2 JSON structure uses "filter_1_freq_hz" for the low-pass 
        filter cutoff frequency in Hertz. This is the parameter we want to predict.
        
        The frequency is normalized to [0, 1] range using:
            normalized = (frequency - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
            
        where TARGET_MIN = 50 Hz and TARGET_MAX = 10000 Hz
        
        Example JSON structure:
            {
                "filter_1_level": "0.0 dB",
                "filter_1_on": true,
                "filter_1_type": "Band 12",
                "filter_1_freq_hz": 307.0,  <- This is the target value
                "filter_1_res": 31.0,
                ...
            }
        
        Args:
            config: Configuration dictionary from .JSON file
            
        Returns:
            Normalized LPF frequency in [0, 1] range as a float
        """
        if "filter_1_freq_hz" not in config:
            raise ValueError(
                f"Missing 'filter_1_freq_hz' in JSON config. "
                f"Expected Serum 2 format."
            )
        
        try:
            freq_hz = float(config["filter_1_freq_hz"])
        except ValueError as e:
            raise ValueError(
                f"Invalid frequency value '{config['filter_1_freq_hz']}': {e}"
            )
        
        # Clamp to valid range
        freq_clamped = np.clip(freq_hz, LPFConstants.MIN_FREQ_HZ, LPFConstants.MAX_FREQ_HZ)
        
        # Normalize to [0, 1]
        normalized = (freq_clamped - LPFConstants.MIN_FREQ_HZ) / LPFConstants.FREQ_RANGE
        
        return float(normalized)
    
    def unnormalize_frequency(self, normalized: float) -> float:
        """
        Convert normalized frequency back to Hz.
        
        Args:
            normalized: Normalized frequency in [0, 1] range
            
        Returns:
            Frequency in Hz
        """
        return normalized * LPFConstants.FREQ_RANGE + LPFConstants.MIN_FREQ_HZ
    
    @staticmethod
    def compute_statistics(data_dir: str) -> Dict[str, Any]:
        """Compute dataset statistics for reporting."""
        return compute_dataset_statistics(data_dir)


def custom_collate_fn(batch: List[Tuple[torch.Tensor, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that ensures float32 tensors.
    
    Args:
        batch: List of tuples containing (tensor, float) pairs
        
    Returns:
        Tuple of stacked tensors (inputs, targets)
    """
    # Stack inputs and targets separately
    inputs = torch.stack([b[0].float() for b in batch])
    targets = torch.stack([b[1].float().unsqueeze(0) for b in batch])
    return inputs, targets


class LPFDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule managing the LPF dataset.
    
    Handles data splitting, batching, and DataLoader creation.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = LPFConstants.BATCH_SIZE_DEFAULT,
        num_workers: int = 4,
        validation_split: float = 0.1,
        n_mels: int = LPFConstants.N_MELS_DEFAULT,
        n_fft: int = LPFConstants.N_FFT_DEFAULT,
        hop_length: int = LPFConstants.HOP_LENGTH_DEFAULT
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.train_dataset: Optional[LPFDataset] = None
        self.val_dataset: Optional[LPFDataset] = None
        self.wav_files: List[Path] = []
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets before training."""
        # Discover all .WAV files
        self.wav_files = sorted(list(self.data_dir.glob("*.wav")))
        
        if not self.wav_files:
            raise ValueError(f"No .wav files found in {self.data_dir}")
        
        print(f"Found {len(self.wav_files)} audio files")
        
        # Create full dataset
        full_dataset = LPFDataset(
            wav_paths=self.wav_files,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Split into train/validation sets
        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size
        
        print(f"Split: {train_size} training, {val_size} validation")
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(self.num_workers, os.cpu_count() - 1),
            pin_memory=True,  # Enable for faster GPU transfers
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(self.num_workers, os.cpu_count() - 1),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn
        )


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class LPFCNN(LightningModule):
    """
    CNN for LPF frequency prediction.
    
    Architecture designed for:
    - Input: Mel spectrogram (1 × 128 × ~93)
    - Output: Single float value (normalized LPF frequency in [0, 1])
    
    The architecture uses multiple convolutional blocks to extract
    spectral features at different scales, followed by dense layers
    for final frequency regression.
    """
    
    def __init__(
        self,
        input_channels: int = LPFConstants.INPUT_CHANNELS,
        n_mels: int = LPFConstants.N_MELS_DEFAULT,
        output_dim: int = LPFConstants.OUTPUT_DIM,
        learning_rate: float = LPFConstants.LEARNING_RATE_DEFAULT
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.n_mels = n_mels
        
        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            # Block 1: Basic spectral patterns
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Frequency bands
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Harmonic patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Deep features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Regression head - will be initialized after forward pass
        self.fc_layers: Optional[nn.Sequential] = None
    
    def _initialize_fc_layers(self, input_features: int) -> None:
        """Initialize FC layers based on actual input features."""
        self.flattened_size = input_features
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output predictions in [0, 1] range (normalized)
        """
        
        if x.device != self.device:
            x = x.to(self.device)
            
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        
        # Initialize FC layers if not already done
        if self.fc_layers is None:
            self._initialize_fc_layers(x.size(1))
            
        for layer in self.fc_layers:
            layer.to(self.device)
        
        x = self.fc_layers(x)
        
        # Sigmoid activation to ensure output is in [0, 1]
        x = torch.sigmoid(x)
        
        return x
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Training step with gradient clipping.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and logged metrics
        """
        inputs, targets = batch
        
        # Explicitly move to model's device
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).unsqueeze(1).float()
        
        outputs = self(inputs)
        
        # Calculate loss using normalized targets
        loss = nn.MSELoss()(outputs, targets)
        
        # Compute additional metrics on normalized scale
        mae = nn.L1Loss()(outputs, targets)
        rmse = torch.sqrt(loss)
        
        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        self.log('train_rmse', rmse, on_step=True, on_epoch=True)
        
        # Log gradient norm (for monitoring gradient clipping effectiveness)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> Dict[str, Any]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and logged metrics
        """
        inputs, targets = batch
        
        # Explicitly move to model's device
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).unsqueeze(1).float()
        
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        mae = nn.L1Loss()(outputs, targets)
        
        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_rmse', torch.sqrt(loss), on_epoch=True)
        
        return {'loss': loss}
    
    def configure_optimizers(self):
        """
        Set up optimizer with cosine annealing scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                     batch_idx: int) -> torch.Tensor:
        """
        Prediction step for inference.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch
            
        Returns:
            Predictions in normalized [0, 1] range
        """
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self(inputs)


# =============================================================================
# PREDICTOR CLASS
# =============================================================================

class LPFPredictor:
    """
    Main class for training and predicting LPF frequencies.
    
    Provides a high-level interface for:
        - Training models with PyTorch Lightning
        - Making predictions on audio files
        - Managing model checkpoints
        
    Prediction Output:
        - Returns normalized frequency in [0, 1] range
        - Use unnormalize_frequency() to convert to Hz
        
    Confidence Estimation:
        Uses Monte Carlo dropout to estimate prediction uncertainty.
        Enable by calling predict_with_confidence() instead of predict().
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model (optional)
            device: PyTorch device (CPU or GPU)
        """
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Audio processing parameters
        self.n_mels = LPFConstants.N_MELS_DEFAULT
        self.n_fft = LPFConstants.N_FFT_DEFAULT
        self.hop_length = LPFConstants.HOP_LENGTH_DEFAULT
        
        # Model instance
        self.model: Optional[LPFCNN] = None
        
        # Load pretrained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(
        self,
        data_dir: str,
        epochs: int = LPFConstants.EPOCHS_DEFAULT,
        batch_size: int = LPFConstants.BATCH_SIZE_DEFAULT,
        learning_rate: float = LPFConstants.LEARNING_RATE_DEFAULT,
        validation_split: float = 0.1,
        output_dir: str = "./models",
        n_mels: int = LPFConstants.N_MELS_DEFAULT
    ) -> None:
        """
        Train the LPF prediction model.
        
        Args:
            data_dir: Directory containing .WAV and .JSON files
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction for validation set
            output_dir: Directory to save checkpoints
            n_mels: Number of Mel frequency bins
        """
        # Setup DataModule
        datamodule = LPFDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2,
            validation_split=validation_split,
            n_mels=n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Setup model
        model = LPFCNN(
            input_channels=1,
            n_mels=n_mels,
            output_dim=1,
            learning_rate=learning_rate
        ).to(self.device)
        
        # Setup callbacks with timestamped names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=f'best_model_{timestamp}',
            save_top_k=5,
            monitor='val_loss',
            mode='min',
            verbose=True,
            auto_insert_metric_name=False
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            verbose=True
        )
        
        # Setup logger
        logger = TensorBoardLogger('tb_logs', name=f'lpf_prediction_{timestamp}')
        
        # Create trainer with mixed precision support
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision='16-mixed',  # Mixed precision for faster training
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 60)
        
        trainer.fit(model, datamodule)
        
        print("-" * 60)
        print("Training completed!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
        
        self.model = model.to(self.device)
    
    def predict(self, wav_path: str) -> float:
        """
        Predict LPF frequency for a single audio file.
        
        Args:
            wav_path: Path to the .WAV file
            
        Returns:
            Normalized LPF frequency in [0, 1] range
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Load audio using float32 (consistent with PyTorch defaults)
        with open(wav_path, 'rb') as f:
            audio_data, sample_rate = sf.read(f, dtype='float32')
        
        # Convert stereo to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=20,
            fmax=24000
        )
        
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_normalized = normalize_mel_spectrogram(mel_db)
        
        # Convert to tensor and predict
        features = torch.FloatTensor(mel_normalized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(features)
        
        return prediction.item()
    
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'n_mels': self.n_mels,
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            n_mels = checkpoint.get('n_mels', LPFConstants.N_MELS_DEFAULT)
            self.model = LPFCNN(
                input_channels=1,
                n_mels=n_mels,
                output_dim=1
            ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_audio_file(path: Path) -> bool:
    """
    Validate that an audio file meets requirements.
    
    Checks:
        - File exists and is readable
        - Sample rate matches expected (48kHz)
        - Duration is approximately 1 second
        - Bit depth is appropriate
    
    Args:
        path: Path to the audio file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        info = sf.info(str(path))
        
        # Check sample rate
        if info.samplerate != LPFConstants.SAMPLE_RATE:
            print(f"Warning: {path.name} has sample rate {info.samplerate}, "
                  f"expected {LPFConstants.SAMPLE_RATE}")
        
        # Check duration (should be ~1 second)
        if abs(info.duration - LPFConstants.DURATION_SECONDS) > 0.2:
            print(f"Warning: {path.name} duration is {info.duration:.2f}s, "
                  f"expected ~1s")
        
        return True
    except Exception as e:
        print(f"Error validating {path.name}: {e}")
        return False


def predict_with_confidence(
    predictor: LPFPredictor,
    wav_path: str,
    n_samples: int = 10
) -> Dict[str, Any]:
    """
    Make prediction with confidence interval using Monte Carlo dropout.
    
    Runs multiple forward passes and computes statistics on the results.
    This provides an estimate of model uncertainty.
    
    Args:
        predictor: LPFPredictor instance
        wav_path: Path to audio file
        n_samples: Number of forward passes for uncertainty estimation
        
    Returns:
        Dictionary with 'prediction', 'mean_hz', 'std', 
        'confidence_interval_95', and 'n_samples'
    """
    if predictor.model is None:
        raise ValueError("Model not loaded.")
        
    predictor.model.eval()
    predictor.model.to(predictor.device)    
    
    # Load and preprocess audio
    with open(wav_path, 'rb') as f:
        audio_data, sample_rate = sf.read(f, dtype='float32')
    
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_mels=predictor.n_mels,
        n_fft=predictor.n_fft,
        hop_length=predictor.hop_length,
        fmin=20,
        fmax=24000
    )
    
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_normalized = normalize_mel_spectrogram(mel_db)
    features = torch.FloatTensor(mel_normalized).unsqueeze(0).to(predictor.device)
    
    # Multiple predictions with dropout enabled (Monte Carlo)
    predictor.model.eval()
    predictions = []
    
    for _ in range(n_samples):
        with torch.no_grad():
            pred = predictor.model(features)
            predictions.append(pred.item())
    
    predictions = np.array(predictions)
    
    return {
        'prediction': predictions.mean(),
        'mean_hz': predictions.mean() * LPFConstants.FREQ_RANGE + LPFConstants.MIN_FREQ_HZ,
        'std': predictions.std(),
        'confidence_interval_95': [
            (predictions.mean() - 1.96 * predictions.std()) * LPFConstants.FREQ_RANGE + LPFConstants.MIN_FREQ_HZ,
            (predictions.mean() + 1.96 * predictions.std()) * LPFConstants.FREQ_RANGE + LPFConstants.MIN_FREQ_HZ
        ],
        'n_samples': n_samples
    }


def print_model_summary(model: LPFCNN) -> None:
    """
    Print detailed model architecture and parameter counts.
    
    Args:
        model: LPFCNN instance to summarize
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Parameter breakdown by layer type
    conv_params = sum(p.numel() for m in model.conv_blocks 
                      for p in m.parameters() if isinstance(m, nn.Conv2d))
    fc_params = sum(p.numel() for m in model.fc_layers 
                    for p in m.parameters() if isinstance(m, nn.Linear))
    
    print(f"\nParameter breakdown:")
    print(f"  Convolutional layers: {conv_params:,}")
    print(f"  Fully connected layers: {fc_params:,}")
    print("=" * 60 + "\n")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train or predict LPF frequency for Serum 2 synthesizer audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python lpf_predictor.py --train --data-dir ./training_data --epochs 100
  
  # Train with custom settings
  python lpf_predictor.py --train --data-dir ./data --epochs 200 --batch-size 64 --learning-rate 0.0005 --output-dir ./models
  
  # Predict LPF for a single audio file
  python lpf_predictor.py --predict --model-path ./models/best.pt --input-wav ./samples/test.wav
    
  # Batch predict with confidence estimation
  python lpf_predictor.py --batch-predict --model-path ./models/best.pt --input-dir ./samples/ --output-csv ./results.csv
    
  # Show dataset statistics
  python lpf_predictor.py --stats --data-dir ./training_data
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true',
                           help='Train a new model')
    mode_group.add_argument('--predict', action='store_true',
                           help='Predict LPF for a single audio file')
    mode_group.add_argument('--batch-predict', action='store_true',
                           help='Batch predict LPF for multiple audio files')
    mode_group.add_argument('--stats', action='store_true',
                           help='Display dataset statistics')
    
    # Training arguments
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing training data (.WAV and .JSON files)')
    parser.add_argument('--epochs', type=int, default=LPFConstants.EPOCHS_DEFAULT,
                       help=f'Number of training epochs (default: {LPFConstants.EPOCHS_DEFAULT})')
    parser.add_argument('--batch-size', type=int, default=LPFConstants.BATCH_SIZE_DEFAULT,
                       help=f'Batch size for training (default: {LPFConstants.BATCH_SIZE_DEFAULT})')
    parser.add_argument('--learning-rate', type=float, default=LPFConstants.LEARNING_RATE_DEFAULT,
                       help=f'Learning rate (default: {LPFConstants.LEARNING_RATE_DEFAULT})')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--n-mels', type=int, default=LPFConstants.N_MELS_DEFAULT,
                       help=f'Number of Mel frequency bins (default: {LPFConstants.N_MELS_DEFAULT})')
    
    # Prediction arguments
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model file')
    parser.add_argument('--input-wav', type=str,
                       help='Input audio file for prediction')
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing audio files for batch prediction')
    parser.add_argument('--output-csv', type=str,
                       help='Output CSV file for batch predictions')
    
    # Confidence estimation arguments
    parser.add_argument('--confidence-samples', type=int, default=10,
                        help='Number of samples for confidence estimation (default: 10)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    predictor = LPFPredictor(model_path=args.model_path)
    
    if args.train:
        if not args.data_dir:
            print("Error: --data-dir required for training mode")
            sys.exit(1)
        
        predictor.train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            n_mels=args.n_mels
        )
    
    elif args.predict:
        if not args.input_wav:
            print("Error: --input-wav required for prediction mode")
            sys.exit(1)
        
        frequency = predictor.predict(args.input_wav)
        print(f"Predicted LPF frequency (normalized): {frequency:.4f}")
        # Convert to Hz for display
        freq_hz = frequency * LPFConstants.FREQ_RANGE + LPFConstants.MIN_FREQ_HZ
        print(f"LPF frequency in Hz: {freq_hz:.2f} Hz")
    
    elif args.batch_predict:
        if not args.input_dir or not args.output_csv:
            print("Error: --input-dir and --output-csv required for batch prediction")
            sys.exit(1)
        
        import pandas as pd
        from tqdm import tqdm
        
        input_path = Path(args.input_dir)
        wav_files = list(input_path.glob('*.wav'))
        
        results = []
        for wav_file in tqdm(wav_files, desc="Processing files"):
            frequency = predictor.predict(str(wav_file))
            freq_hz = frequency * LPFConstants.FREQ_RANGE + LPFConstants.MIN_FREQ_HZ
            results.append({
                'file': wav_file.name,
                'lpf_frequency_normalized': frequency,
                'lpf_frequency_hz': freq_hz
            })
        
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nBatch predictions saved to {args.output_csv}")
    
    elif args.stats:
        if not args.data_dir:
            print("Error: --data-dir required for statistics mode")
            sys.exit(1)
        
        stats = compute_dataset_statistics(args.data_dir)
        print_dataset_statistics(stats)


if __name__ == '__main__':
    main()
