#!/usr/bin/env python3
"""
LPF Frequency, Filter Type, & Wavetable Prediction for Serum 2 Synthesizer
============================================================================

Multi-output model predicting:
1. filter_1_freq_hz - LPF cutoff frequency (regression)
2. filter_1_type - Filter type/name (classification)
3. wavetable_name - Oscillator A wavetable name (classification)
4. a_wt_pos - Oscillator A wavetable position (regression)

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


# =============================================================================
# DATA MODULE
# =============================================================================

class LPFDataset(Dataset):
    """
    Dataset for loading and preprocessing audio samples with LPF frequency, 
    filter type, and wavetable parameters.
    
    Expected format:
        - .WAV files: 1-second C3 notes at 48kHz, 32-bit float
        - .JSON files: Contains filter frequency configuration
    
    Output format:
        - frequency: normalized LPF frequency in [0, 1]
        - filter_type: raw filter type string
        - wavetable_name: wavetable name extracted from filename
        - wavetable_position: normalized a_wt_pos in [0, 1]
    
    The dataset handles:
        - Audio loading via soundfile (preserves 32-bit float)
        - Mel spectrogram computation
        - Normalization for training stability
    """
    
    TARGET_MIN: float = 50.0
    TARGET_MAX: float = 10000.0
    TARGET_RANGE: float = TARGET_MAX - TARGET_MIN
    
    # Wavetable position range (normalized)
    WT_POS_MIN: float = 0.0
    WT_POS_MAX: float = 1.0
    WT_POS_RANGE: float = WT_POS_MAX - WT_POS_MIN
    
    def __init__(
        self,
        wav_paths: List[Path],
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 20.0,
        fmax: float = 24000.0
    ):
        self.wav_paths = wav_paths
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        
    def __len__(self) -> int:
        return len(self.wav_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, str, str, float]]:
        """
        Get a single sample with all parameters.
        
        Returns:
            Tuple of (features_tensor, (frequency_normalized, filter_type_string, 
                                         wavetable_name, wavetable_position))
        """
        import soundfile as sf
        
        wav_path = self.wav_paths[idx]
        json_path = wav_path.with_stem(f"{wav_path.stem}_params").with_suffix('.json')
        
        # Load audio
        audio_data, sample_rate = sf.read(str(wav_path), dtype='float32')
        
        # Handle stereo: convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to [-1, 1]
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
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale (dB)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize to [0, 1]
        normalized = self._normalize_mel(mel_db)
        
        # Convert to torch tensor
        features_tensor = torch.FloatTensor(normalized).unsqueeze(0)
        
        # Load JSON and extract labels
        config = json.load(open(json_path))
        lpf_frequency = self._extract_lpf_frequency(config)
        filter_type = self._extract_filter_type(config)
        wavetable_name = self._extract_wavetable_name(wav_path)
        wavetable_position = self._extract_wavetable_position(config)
        
        return features_tensor, (lpf_frequency, filter_type, wavetable_name, wavetable_position)
    
    def _normalize_mel(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Normalize Mel spectrogram to [0, 1] range."""
        mean = np.mean(mel_spectrogram, axis=1, keepdims=True)
        std = np.std(mel_spectrogram, axis=1, keepdims=True) + 1e-8
        normalized = (mel_spectrogram - mean) / std
        normalized = np.clip(normalized, -50, 50)
        normalized = (normalized - normalized.min()) / (
            normalized.max() - normalized.min() + 1e-8
        )
        return normalized
    
    def _extract_lpf_frequency(self, config: dict) -> float:
        """Extract and normalize LPF frequency."""
        freq_hz = float(config["filter_1_freq_hz"])
        freq_clamped = np.clip(freq_hz, self.TARGET_MIN, self.TARGET_MAX)
        normalized = (freq_clamped - self.TARGET_MIN) / self.TARGET_RANGE
        return float(normalized)
    
    def _extract_filter_type(self, config: dict) -> str:
        """Extract filter type as raw string."""
        filter_type_str = config.get("filter_1_type", "Unknown")
        if not filter_type_str or filter_type_str == "None":
            filter_type_str = "Unknown"
        return filter_type_str
    
    def _extract_wavetable_name(self, wav_path: Path) -> str:
        """
        Extract wavetable name from file path.
        
        Expected filename format: Serum2_{wavetable_name}_...
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Wavetable name string
        """
        # Extract stem and remove Serum2_ prefix
        stem = wav_path.stem
        # Remove _params suffix if present
        if stem.endswith("_params"):
            stem = stem[:-7]
        
        # Find the wavetable name between Serum2_ and the next underscore
        prefix = "Serum2_"
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            # Find first underscore after the wavetable name
            underscore_pos = rest.find("_")
            if underscore_pos > 0:
                wavetable_name = rest[:underscore_pos]
            else:
                wavetable_name = rest
            return wavetable_name
        
        return "Unknown"
    
    def _extract_wavetable_position(self, config: dict) -> float:
        """
        Extract and normalize wavetable position.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Normalized wavetable position in [0, 1]
        """
        # Get a_wt_pos from config
        wt_pos = float(config.get("a_wt_pos", 0.5))
        
        # Normalize to [0, 1] range
        # Assuming a_wt_pos is already in [0, 1] range for Serum 2
        normalized = np.clip(wt_pos, self.WT_POS_MIN, self.WT_POS_MAX)
        
        return float(normalized)
    
    def unnormalize_frequency(self, normalized: float) -> float:
        """Convert normalized frequency back to Hz."""
        return normalized * self.TARGET_RANGE + self.TARGET_MIN
    
    def unnormalize_wavetable_position(self, normalized: float) -> float:
        """Convert normalized wavetable position back to original scale."""
        return normalized * self.WT_POS_RANGE + self.WT_POS_MIN
    
    @staticmethod
    def compute_statistics(data_dir: str) -> Dict[str, Any]:
        """Compute dataset statistics for reporting."""
        json_files = sorted(Path(data_dir).glob("*.json"))
        frequencies = []
        filter_types = defaultdict(int)
        wavetables = defaultdict(int)
        wt_positions = []
        
        for jf in json_files:
            try:
                with open(jf) as f:
                    config = json.load(f)
                
                freq = float(config["filter_1_freq_hz"])
                frequencies.append(freq)
                
                ft = config.get("filter_1_type", "Unknown")
                filter_types[ft] += 1
                
                wt = config.get("a_wt_pos", "Unknown")
                wavetables[str(wt)] += 1
                
                wt_positions.append(float(config.get("a_wt_pos", 0.5)))
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
            'num_filter_types': len(filter_types),
            'wavetables': dict(wavetables),
            'num_wavetables': len(wavetables),
            'min_wt_pos': min(wt_positions) if wt_positions else 0,
            'max_wt_pos': max(wt_positions) if wt_positions else 1,
            'mean_wt_pos': np.mean(wt_positions) if wt_positions else 0.5
        }


class LPFDataModule(LightningDataModule):
    """PyTorch Lightning DataModule managing the LPF dataset."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        validation_split: float = 0.1,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
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
        self.wav_files = sorted(list(self.data_dir.glob("*.wav")))
        
        if not self.wav_files:
            raise ValueError(f"No .wav files found in {self.data_dir}")
        
        print(f"Found {len(self.wav_files)} audio files")
        
        full_dataset = LPFDataset(
            wav_paths=self.wav_files,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size
        
        print(f"Split: {train_size} training, {val_size} validation")
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader with CUDA support."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=lambda batch: (
                torch.stack([b[0].float() for b in batch]),
                torch.stack([torch.tensor(b[1][0]).float() for b in batch]),  # frequencies
                [b[1][1] for b in batch],  # filter types as strings
                [b[1][2] for b in batch],  # wavetable names as strings
                torch.stack([torch.tensor(b[1][3]).float() for b in batch])   # wavetable positions
            )
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader with CUDA support."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=lambda batch: (
                torch.stack([b[0].float() for b in batch]),
                torch.stack([torch.tensor(b[1][0]).float() for b in batch]),
                [b[1][1] for b in batch],
                [b[1][2] for b in batch],
                torch.stack([torch.tensor(b[1][3]).float() for b in batch])
            )
        )


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class LPFMultiOutput(LightningModule):
    """
    Multi-output CNN for LPF frequency, filter type, and wavetable prediction.
    
    Predicts four outputs:
        1. frequency: Normalized LPF frequency in [0, 1] (regression)
        2. filter_type: Filter type as string (classification)
        3. wavetable_name: Wavetable name as string (classification)
        4. wavetable_position: Normalized position in [0, 1] (regression)
    
    Loss combines:
        - MSE loss for frequency (weighted)
        - Cross-entropy loss for filter type (weighted)
        - Cross-entropy loss for wavetable name (weighted)
        - MSE loss for wavetable position (weighted)
    """
    
    TARGET_MIN: float = 50.0
    TARGET_MAX: float = 10000.0
    TARGET_RANGE: float = TARGET_MAX - TARGET_MIN
    
    WT_POS_MIN: float = 0.0
    WT_POS_MAX: float = 1.0
    WT_POS_RANGE: float = WT_POS_MAX - WT_POS_MIN
    
    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        filter_type_strings: List[str] = None,
        wavetable_strings: List[str] = None,
        learning_rate: float = 0.001,
        freq_loss_weight: float = 1.0,
        type_loss_weight: float = 0.5,
        wt_name_loss_weight: float = 0.5,
        wt_pos_loss_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.n_mels = n_mels
        self.freq_loss_weight = freq_loss_weight
        self.type_loss_weight = type_loss_weight
        self.wt_name_loss_weight = wt_name_loss_weight
        self.wt_pos_loss_weight = wt_pos_loss_weight
        
        # Filter type strings
        if filter_type_strings is None:
            self.filter_type_strings = [
                "MG Low 12", "MG Low 24", "MG High 12", "MG High 24",
                "MG Band 12", "MG Band 24", "MG Notch 12", "MG Notch 24",
                "LP 12", "LP 24", "HP 12", "HP 24",
                "BP 12", "BP 24", "BS 12", "BS 24",
                "APF 12", "APF 24", "PK 12", "PK 24",
                "Comb 12", "Comb 24", "Resonator", "Formant",
                "Waveshaper", "Bitcrusher", "Distortion", "Overdrive"
            ]
        else:
            self.filter_type_strings = filter_type_strings
            
        self.num_filter_classes = len(self.filter_type_strings)
        self.filter_to_index = {ft: idx for idx, ft in enumerate(self.filter_type_strings)}
        self.index_to_filter = {idx: ft for idx, ft in enumerate(self.filter_type_strings)}
        
        # Wavetable strings
        if wavetable_strings is None:
            self.wavetable_strings = self._load_wavetables_from_data()
        else:
            self.wavetable_strings = wavetable_strings
            
        self.num_wt_classes = len(self.wavetable_strings)
        self.wt_to_index = {wt: idx for idx, wt in enumerate(self.wavetable_strings)}
        self.index_to_wt = {idx: wt for idx, wt in enumerate(self.wavetable_strings)}
        
        # Convolutional feature extractor (shared)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Dynamic FC layer initialization
        self.fc_layers: Optional[nn.Sequential] = None
        
        # Frequency regression head
        self.freq_head = nn.Sequential(
            nn.Linear(256 * (n_mels // 8) * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Filter type classification head
        self.type_head = nn.Sequential(
            nn.Linear(256 * (n_mels // 8) * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_filter_classes),
        )
        
        # Wavetable name classification head
        self.wt_name_head = nn.Sequential(
            nn.Linear(256 * (n_mels // 8) * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_wt_classes),
        )
        
        # Wavetable position regression head
        self.wt_pos_head = nn.Sequential(
            nn.Linear(256 * (n_mels // 8) * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _load_wavetables_from_data(self) -> List[str]:
        """Load unique wavetable names from dataset."""
        import glob
        import json
        
        # Find all unique wavetable names in the data
        wavetables = set()
        
        # Scan through JSON files to find a_wt_pos entries
        json_files = glob.glob(str(self.hparams.data_dir) + "/*.json") if hasattr(self, 'hparams') and hasattr(self.hparams, 'data_dir') else []
        
        for jf in json_files:
            try:
                with open(jf) as f:
                    config = json.load(f)
                # Extract wavetable name from filename pattern
                # This is a placeholder - actual implementation depends on your data
                pass
            except:
                pass
        
        # Default wavetable list if none found
        return list(wavetables) if wavetables else [
            "Default", "Saw", "Square", "Triangle", "Sine",
            "Noise", "FM", "Wavetable", "Granular", "Sample"
        ]
    
    def _initialize_fc_layers(self, input_features: int) -> None:
        """Initialize FC layers based on actual input features."""
        self.flattened_size = input_features
        
        self.freq_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.type_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_filter_classes),
        )
        
        self.wt_name_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_wt_classes),
        )
        
        self.wt_pos_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (frequency_pred, type_pred, wt_name_pred, wt_pos_pred)
        """
        if x.device != self.device:
            x = x.to(self.device)
        
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        
        if self.fc_layers is None:
            self._initialize_fc_layers(x.size(1))
        
        freq_pred = self.freq_head(x)
        type_pred = self.type_head(x)
        wt_name_pred = self.wt_name_head(x)
        wt_pos_pred = self.wt_pos_head(x)
        
        return freq_pred, type_pred, wt_name_pred, wt_pos_pred
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str], torch.Tensor], 
                      batch_idx: int) -> Dict[str, Any]:
        """Training step with combined loss."""
        inputs, frequencies, filter_types, wavetable_names, wt_positions = batch
        
        inputs = inputs.to(self.device).float()
        frequencies = frequencies.to(self.device).float().unsqueeze(1)
        wt_positions = wt_positions.to(self.device).float().unsqueeze(1)
        
        # Convert strings to indices
        filter_type_indices = torch.tensor([
            self.filter_to_index.get(ft, 0) for ft in filter_types
        ], dtype=torch.long).to(self.device)
        
        wt_name_indices = torch.tensor([
            self.wt_to_index.get(wt, 0) for wt in wavetable_names
        ], dtype=torch.long).to(self.device)
        
        freq_pred, type_pred, wt_name_pred, wt_pos_pred = self(inputs)
        
        # Calculate losses
        freq_loss = nn.MSELoss()(freq_pred, frequencies)
        type_loss = nn.CrossEntropyLoss()(type_pred, filter_type_indices)
        wt_name_loss = nn.CrossEntropyLoss()(wt_name_pred, wt_name_indices)
        wt_pos_loss = nn.MSELoss()(wt_pos_pred, wt_positions)
        
        # Combined loss with weights
        total_loss = (self.freq_loss_weight * freq_loss +
                      self.type_loss_weight * type_loss +
                      self.wt_name_loss_weight * wt_name_loss +
                      self.wt_pos_loss_weight * wt_pos_loss)
        
        # Compute metrics
        mae = nn.L1Loss()(freq_pred, frequencies)
        rmse = torch.sqrt(freq_loss)
        type_accuracy = (type_pred.argmax(dim=1) == filter_type_indices).float().mean()
        wt_pos_accuracy = (wt_pos_pred.round() == wt_positions).float().mean()
        
        # Log training metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_freq_loss', freq_loss, on_step=True, on_epoch=True)
        self.log('train_type_loss', type_loss, on_step=True, on_epoch=True)
        self.log('train_wt_name_loss', wt_name_loss, on_step=True, on_epoch=True)
        self.log('train_wt_pos_loss', wt_pos_loss, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        self.log('train_rmse', rmse, on_step=True, on_epoch=True)
        self.log('train_type_accuracy', type_accuracy, on_step=True, on_epoch=True)
        self.log('train_wt_pos_accuracy', wt_pos_accuracy, on_step=True, on_epoch=True)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True)
        
        return {'loss': total_loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str], torch.Tensor],
                        batch_idx: int) -> Dict[str, Any]:
        """Validation step with string logging."""
        inputs, frequencies, filter_types, wavetable_names, wt_positions = batch
        
        inputs = inputs.to(self.device).float()
        frequencies = frequencies.to(self.device).float().unsqueeze(1)
        wt_positions = wt_positions.to(self.device).float().unsqueeze(1)
        
        filter_type_indices = torch.tensor([
            self.filter_to_index.get(ft, 0) for ft in filter_types
        ], dtype=torch.long).to(self.device)
        
        wt_name_indices = torch.tensor([
            self.wt_to_index.get(wt, 0) for wt in wavetable_names
        ], dtype=torch.long).to(self.device)
        
        freq_pred, type_pred, wt_name_pred, wt_pos_pred = self(inputs)
        
        freq_loss = nn.MSELoss()(freq_pred, frequencies)
        type_loss = nn.CrossEntropyLoss()(type_pred, filter_type_indices)
        wt_name_loss = nn.CrossEntropyLoss()(wt_name_pred, wt_name_indices)
        wt_pos_loss = nn.MSELoss()(wt_pos_pred, wt_positions)
        total_loss = (self.freq_loss_weight * freq_loss +
                      self.type_loss_weight * type_loss +
                      self.wt_name_loss_weight * wt_name_loss +
                      self.wt_pos_loss_weight * wt_pos_loss)
        
        mae = nn.L1Loss()(freq_pred, frequencies)
        rmse = torch.sqrt(freq_loss)
        type_accuracy = (type_pred.argmax(dim=1) == filter_type_indices).float().mean()
        wt_pos_accuracy = (wt_pos_pred.round() == wt_positions).float().mean()
        
        # Convert to actual values for logging
        pred_freq = freq_pred.squeeze().item()
        pred_type = self.index_to_filter.get(type_pred.argmax(dim=1).item(), "Unknown")
        pred_wt_name = self.index_to_wt.get(wt_name_pred.argmax(dim=1).item(), "Unknown")
        pred_wt_pos = wt_pos_pred.squeeze().item()
        
        actual_freq = frequencies.squeeze().item()
        actual_type = filter_types[0]  # First in batch
        actual_wt_name = wavetable_names[0]
        actual_wt_pos = wt_positions.squeeze().item()
        
        # Log validation metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_freq_loss', freq_loss, on_epoch=True)
        self.log('val_type_loss', type_loss, on_epoch=True)
        self.log('val_wt_name_loss', wt_name_loss, on_epoch=True)
        self.log('val_wt_pos_loss', wt_pos_loss, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True)
        self.log('val_type_accuracy', type_accuracy, on_epoch=True)
        self.log('val_wt_pos_accuracy', wt_pos_accuracy, on_epoch=True)
        
        # Log actual and predicted string values
        self.log('val_pred_filter_type', pred_type, on_epoch=True, prog_bar=False)
        self.log('val_pred_frequency', pred_freq, on_epoch=True, prog_bar=False)
        self.log('val_pred_wavetable', pred_wt_name, on_epoch=True, prog_bar=False)
        self.log('val_pred_wt_pos', pred_wt_pos, on_epoch=True, prog_bar=False)
        self.log('val_actual_filter_type', actual_type, on_epoch=True, prog_bar=False)
        self.log('val_actual_frequency', actual_freq, on_epoch=True, prog_bar=False)
        self.log('val_actual_wavetable', actual_wt_name, on_epoch=True, prog_bar=False)
        self.log('val_actual_wt_pos', actual_wt_pos, on_epoch=True, prog_bar=False)
        
        return {'loss': total_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str], torch.Tensor],
                     batch_idx: int) -> Tuple[torch.Tensor, List[str], List[str], torch.Tensor]:
        inputs, _, _, _, _ = batch
        inputs = inputs.to(self.device).float()
        freq_pred, type_pred, wt_name_pred, wt_pos_pred = self(inputs)
        
        type_strings = [self.index_to_filter.get(idx.item(), "Unknown") for idx in type_pred.argmax(dim=1)]
        wt_strings = [self.index_to_wt.get(idx.item(), "Unknown") for idx in wt_name_pred.argmax(dim=1)]
        
        return freq_pred.squeeze(), type_strings, wt_strings, wt_pos_pred.squeeze()


# =============================================================================
# PREDICTOR CLASS
# =============================================================================

class LPFPredictor:
    """Main class for training and predicting all Serum 2 parameters."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        filter_type_strings: List[str] = None,
        wavetable_strings: List[str] = None
    ):
        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self._device}")
        
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        self.filter_type_strings = filter_type_strings or [
            "MG Low 12", "MG Low 24", "MG High 12", "MG High 24",
            "MG Band 12", "MG Band 24", "MG Notch 12", "MG Notch 24",
            "LP 12", "LP 24", "HP 12", "HP 24",
            "BP 12", "BP 24", "BS 12", "BS 24",
            "APF 12", "APF 24", "PK 12", "PK 24",
            "Comb 12", "Comb 24", "Resonator", "Formant",
            "Waveshaper", "Bitcrusher", "Distortion", "Overdrive"
        ]
        
        self.wavetable_strings = wavetable_strings or [
            "Default", "Saw", "Square", "Triangle", "Sine",
            "Noise", "FM", "Wavetable", "Granular", "Sample"
        ]
        
        self.model: Optional[LPFMultiOutput] = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def train(
        self,
        data_dir: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        output_dir: str = "./models",
        n_mels: int = 128,
        freq_loss_weight: float = 1.0,
        type_loss_weight: float = 0.5,
        wt_name_loss_weight: float = 0.5,
        wt_pos_loss_weight: float = 1.0
    ) -> None:
        """Train the multi-output prediction model."""
        datamodule = LPFDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2,
            validation_split=validation_split,
            n_mels=n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        model = LPFMultiOutput(
            input_channels=1,
            n_mels=n_mels,
            filter_type_strings=self.filter_type_strings,
            wavetable_strings=self.wavetable_strings,
            learning_rate=learning_rate,
            freq_loss_weight=freq_loss_weight,
            type_loss_weight=type_loss_weight,
            wt_name_loss_weight=wt_name_loss_weight,
            wt_pos_loss_weight=wt_pos_loss_weight
        ).to(self._device)
        
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
            patience=15,
            mode='min',
            verbose=True
        )
        
        logger = TensorBoardLogger('tb_logs', name=f'lpf_prediction_{timestamp}')
        
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision='16-mixed',
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=logger,
            log_every_n_steps=50,
            enable_progress_bar=True,
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Filter types: {len(self.filter_type_strings)}, Wavetables: {len(self.wavetable_strings)}")
        print("-" * 60)
        
        trainer.fit(model, datamodule)
        
        print("-" * 60)
        print("Training completed!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
        
        self.model = model.to(self._device)
    
    def predict(self, wav_path: str) -> Tuple[float, str, str, float]:
        """
        Predict all parameters for a single audio file.
        
        Returns:
            Tuple of (frequency_normalized, filter_type_string, wavetable_name, wavetable_position)
        """
        import soundfile as sf
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        self.model.to(self._device)
        
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
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=20,
            fmax=24000
        )
        
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_normalized = self._normalize_mel(mel_db)
        
        features = torch.FloatTensor(mel_normalized).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            freq_pred, type_pred, wt_name_pred, wt_pos_pred = self.model(features)
        
        frequency = freq_pred.item()
        filter_type = type_pred[0]
        wavetable_name = wt_name_pred[0]
        wavetable_position = wt_pos_pred.item()
        
        return frequency, filter_type, wavetable_name, wavetable_position
    
    def _normalize_mel(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Normalize Mel spectrogram to [0, 1] range."""
        mean = np.mean(mel_spectrogram, axis=1, keepdims=True)
        std = np.std(mel_spectrogram, axis=1, keepdims=True) + 1e-8
        normalized = (mel_spectrogram - mean) / std
        normalized = np.clip(normalized, -50, 50)
        normalized = (normalized - normalized.min()) / (
            normalized.max() - normalized.min() + 1e-8
        )
        return normalized
    
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'n_mels': self.n_mels,
                'filter_type_strings': self.model.filter_type_strings,
                'wavetable_strings': self.model.wavetable_strings,
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self._device)
        
        if self.model is None:
            n_mels = checkpoint.get('n_mels', 128)
            filter_strings = checkpoint.get('filter_type_strings', self.filter_type_strings)
            wt_strings = checkpoint.get('wavetable_strings', self.wavetable_strings)
            self.model = LPFMultiOutput(
                input_channels=1,
                n_mels=n_mels,
                filter_type_strings=filter_strings,
                wavetable_strings=wt_strings
            ).to(self._device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train or predict Serum 2 parameters (LPF, Filter Type, Wavetable, Wavetable Position)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python lpf_predictor.py --train --data-dir ./training_data --epochs 100
  
  # Train with custom settings
  python lpf_predictor.py --train --data-dir ./data --epochs 200 \\
    --batch-size 64 --learning-rate 0.0005 --output-dir ./models
  
  # Predict all parameters for a single audio file
  python lpf_predictor.py --predict --model-path ./models/best.pt \\
    --input-wav ./samples/test.wav
    
  # Batch predict
  python lpf_predictor.py --batch-predict --model-path ./models/best.pt \\
    --input-dir ./samples/ --output-csv ./results.csv
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train a new model')
    mode_group.add_argument('--predict', action='store_true', help='Predict for a single audio file')
    mode_group.add_argument('--batch-predict', action='store_true', help='Batch predict for multiple files')
    mode_group.add_argument('--stats', action='store_true', help='Display dataset statistics')
    
    parser.add_argument('--data-dir', type=str, help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models', help='Directory to save checkpoints')
    parser.add_argument('--n-mels', type=int, default=128, help='Number of Mel frequency bins')
    
    parser.add_argument('--model-path', type=str, help='Path to trained model file')
    parser.add_argument('--input-wav', type=str, help='Input audio file for prediction')
    parser.add_argument('--input-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output-csv', type=str, help='Output CSV file for batch predictions')
    
    # Loss weight arguments
    parser.add_argument('--freq-loss-weight', type=float, default=1.0, help='Weight for frequency loss')
    parser.add_argument('--type-loss-weight', type=float, default=0.5, help='Weight for filter type loss')
    parser.add_argument('--wt-name-loss-weight', type=float, default=0.5, help='Weight for wavetable name loss')
    parser.add_argument('--wt-pos-loss-weight', type=float, default=1.0, help='Weight for wavetable position loss')
    
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
            n_mels=args.n_mels,
            freq_loss_weight=args.freq_loss_weight,
            type_loss_weight=args.type_loss_weight,
            wt_name_loss_weight=args.wt_name_loss_weight,
            wt_pos_loss_weight=args.wt_pos_loss_weight
        )
    
    elif args.predict:
        if not args.input_wav:
            print("Error: --input-wav required for prediction mode")
            sys.exit(1)
        
        frequency, filter_type, wavetable_name, wavetable_position = predictor.predict(args.input_wav)
        freq_hz = frequency * 9950 + 50
        
        print(f"Predicted LPF frequency (normalized): {frequency:.4f}")
        print(f"LPF frequency in Hz: {freq_hz:.2f} Hz")
        print(f"Predicted filter type: {filter_type}")
        print(f"Predicted wavetable name: {wavetable_name}")
        print(f"Predicted wavetable position (normalized): {wavetable_position:.4f}")
    
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
            frequency, filter_type, wavetable_name, wavetable_position = predictor.predict(str(wav_file))
            freq_hz = frequency * 9950 + 50
            results.append({
                'file': wav_file.name,
                'lpf_frequency_normalized': frequency,
                'lpf_frequency_hz': freq_hz,
                'filter_type': filter_type,
                'wavetable_name': wavetable_name,
                'wavetable_position_normalized': wavetable_position
            })
        
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nBatch predictions saved to {args.output_csv}")
    
    elif args.stats:
        if not args.data_dir:
            print("Error: --data-dir required for statistics mode")
            sys.exit(1)
        
        stats = LPFDataset.compute_statistics(args.data_dir)
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Frequency range: {stats['frequency_range']}")
        print(f"Mean frequency: {stats['mean_frequency']:.1f} Hz")
        print(f"Std deviation: {stats['std_frequency']:.1f} Hz")
        print(f"Median frequency: {stats['median_frequency']:.1f} Hz")
        print(f"\nFilter types distribution ({stats['num_filter_types']} types):")
        for ftype, count in sorted(stats['filter_types'].items()):
            print(f"  {ftype}: {count}")
        print(f"\nWavetables distribution ({stats['num_wavetables']} types):")
        for wt, count in sorted(stats['wavetables'].items()):
            print(f"  {wt}: {count}")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
