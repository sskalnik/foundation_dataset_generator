#!/usr/bin/env python3
"""
LPF Frequency & Filter Type Prediction for Serum 2 Synthesizer
==============================================================

Multi-output model predicting:
1. filter_1_freq_hz - LPF cutoff frequency (regression)
2. filter_1_type - Filter type/name (classification)

Author: Assistant
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
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
    Dataset for loading audio samples with LPF frequency and filter type.
    
    Output format:
        - frequency: normalized LPF frequency in [0, 1]
        - filter_type: encoded filter type class index
    """
    
    TARGET_MIN: float = 50.0
    TARGET_MAX: float = 10000.0
    TARGET_RANGE: float = TARGET_MAX - TARGET_MIN
    
    # Filter type mapping - adjust based on your actual data
    FILTER_TYPES = [
        "MG Low 12", "MG Low 24", "MG High 12", "MG High 24",
        "MG Band 12", "MG Band 24", "MG Notch 12", "MG Notch 24",
        "LP 12", "LP 24", "HP 12", "HP 24",
        "BP 12", "BP 24", "BS 12", "BS 24",
        "APF 12", "APF 24", "PK 12", "PK 24",
        "Comb 12", "Comb 24", "Resonator", "Formant",
        "Waveshaper", "Bitcrusher", "Distortion", "Overdrive"
    ]
    
    # Create mappings
    FILTER_TO_INDEX = {ft: idx for idx, ft in enumerate(FILTER_TYPES)}
    INDEX_TO_FILTER = {idx: ft for idx, ft in enumerate(FILTER_TYPES)}
    NUM_CLASSES = len(FILTER_TYPES)
    
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, int]]:
        """
        Get a single sample with frequency and filter type.
        
        Returns:
            Tuple of (features_tensor, (frequency_normalized, filter_type_index))
        """
        import soundfile as sf
        
        wav_path = self.wav_paths[idx]
        json_path = wav_path.with_suffix('.json')
        
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
        filter_type_index = self._extract_filter_type(config)
        
        return features_tensor, (lpf_frequency, filter_type_index)
    
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
    
    def _extract_filter_type(self, config: dict) -> int:
        """
        Extract filter type and return encoded index.
        
        Serum 2 uses various filter type strings. We map these to indices.
        
        Common formats in Serum 2:
            - "MG Low 12" (Multisource Gradient)
            - "LP 12" (Low Pass)
            - "HP 12" (High Pass)
            - "BP 12" (Band Pass)
            - "BS 12" (Band Stop/Notch)
            - "APF 12" (All Pass)
            - "PK 12" (Peak/Parametric)
            - "Comb 12" (Comb Filter)
            - "Resonator" / "Formant"
            - "Waveshaper", "Bitcrusher", etc.
        
        Args:
            config: Configuration dictionary from .JSON file
            
        Returns:
            Integer index representing the filter type
        """
        filter_type_str = config.get("filter_1_type", "Unknown")
        
        # Handle cases where filter type might be None or empty
        if not filter_type_str or filter_type_str == "None":
            filter_type_str = "Unknown"
        
        # Map to index, use 0 (Unknown) if not found
        return self.FILTER_TO_INDEX.get(filter_type_str, 0)
    
    def unnormalize_frequency(self, normalized: float) -> float:
        """Convert normalized frequency back to Hz."""
        return normalized * self.TARGET_RANGE + self.TARGET_MIN
    
    @staticmethod
    def compute_statistics(data_dir: str) -> Dict[str, Any]:
        """Compute dataset statistics for reporting."""
        json_files = sorted(Path(data_dir).glob("*.json"))
        frequencies = []
        filter_types = defaultdict(int)
        
        for jf in json_files:
            try:
                with open(jf) as f:
                    config = json.load(f)
                
                freq = float(config["filter_1_freq_hz"])
                frequencies.append(freq)
                
                ft = config.get("filter_1_type", "Unknown")
                filter_types[ft] += 1
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
            'num_filter_types': len(filter_types)
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
        print(f"Number of filter types: {LPFDataset.NUM_CLASSES}")
        
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
            persistent_workers=False,
            collate_fn=lambda batch: (
                torch.stack([b[0].float() for b in batch]),
                torch.stack([torch.tensor(b[1][0]).float() for b in batch]),  # frequencies
                torch.stack([torch.tensor(b[1][1]).long() for b in batch])    # filter types
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
            persistent_workers=False,
            collate_fn=lambda batch: (
                torch.stack([b[0].float() for b in batch]),
                torch.stack([torch.tensor(b[1][0]).float() for b in batch]),
                torch.stack([torch.tensor(b[1][1]).long() for b in batch])
            )
        )


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class LPFMultiOutput(LightningModule):
    """
    Multi-output CNN for LPF frequency and filter type prediction.
    
    Predicts two outputs:
        1. frequency: Normalized LPF frequency in [0, 1] (regression)
        2. filter_type: Filter type class index (classification)
    
    Loss combines:
        - MSE loss for frequency (weighted)
        - Cross-entropy loss for filter type (weighted)
    """
    
    TARGET_MIN: float = 50.0
    TARGET_MAX: float = 10000.0
    TARGET_RANGE: float = TARGET_MAX - TARGET_MIN
    
    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        num_classes: int = LPFDataset.NUM_CLASSES,
        learning_rate: float = 0.001,
        freq_loss_weight: float = 1.0,
        type_loss_weight: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.freq_loss_weight = freq_loss_weight
        self.type_loss_weight = type_loss_weight
        
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
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Filter type classification head
        self.type_head = nn.Sequential(
            nn.Linear(256 * (n_mels // 8) * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
    
    def _initialize_fc_layers(self, input_features: int) -> None:
        """Initialize FC layers based on actual input features."""
        self.flattened_size = input_features
        
        # Update heads with correct input size
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
            nn.Linear(256, self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (frequency_pred, type_pred)
            - frequency_pred: Normalized frequency in [0, 1]
            - type_pred: Logits for filter type classification
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        
        # Initialize heads if not already done
        if self.fc_layers is None:
            self._initialize_fc_layers(x.size(1))
        
        # Pass through heads
        freq_pred = self.freq_head(x)
        type_pred = self.type_head(x)
        
        return freq_pred, type_pred
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Training step with combined loss.
        
        Args:
            batch: Tuple of (inputs, frequencies, filter_types)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and logged metrics
        """
        inputs, frequencies, filter_types = batch
        
        # Move to device and ensure correct dtypes
        inputs = inputs.to(self.device).float()
        frequencies = frequencies.to(self.device).float().unsqueeze(1)
        filter_types = filter_types.to(self.device).long()
        
        # Forward pass
        freq_pred, type_pred = self(inputs)
        
        # Calculate losses
        freq_loss = nn.MSELoss()(freq_pred, frequencies)
        type_loss = nn.CrossEntropyLoss()(type_pred, filter_types.squeeze())
        
        # Combined loss with weights
        total_loss = self.freq_loss_weight * freq_loss + self.type_loss_weight * type_loss
        
        # Compute metrics
        mae = nn.L1Loss()(freq_pred, frequencies)
        rmse = torch.sqrt(freq_loss)
        type_accuracy = (type_pred.argmax(dim=1) == filter_types.squeeze()).float().mean()
        
        # Log training metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_freq_loss', freq_loss, on_step=True, on_epoch=True)
        self.log('train_type_loss', type_loss, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        self.log('train_rmse', rmse, on_step=True, on_epoch=True)
        self.log('train_type_accuracy', type_accuracy, on_step=True, on_epoch=True)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True)
        
        return {'loss': total_loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (inputs, frequencies, filter_types)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and logged metrics
        """
        inputs, frequencies, filter_types = batch
        
        inputs = inputs.to(self.device).float()
        frequencies = frequencies.to(self.device).float().unsqueeze(1)
        filter_types = filter_types.to(self.device).long()
        
        freq_pred, type_pred = self(inputs)
        
        freq_loss = nn.MSELoss()(freq_pred, frequencies)
        type_loss = nn.CrossEntropyLoss()(type_pred, filter_types.squeeze())
        total_loss = self.freq_loss_weight * freq_loss + self.type_loss_weight * type_loss
        
        mae = nn.L1Loss()(freq_pred, frequencies)
        rmse = torch.sqrt(freq_loss)
        type_accuracy = (type_pred.argmax(dim=1) == filter_types.squeeze()).float().mean()
        
        # Log validation metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_freq_loss', freq_loss, on_epoch=True)
        self.log('val_type_loss', type_loss, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True)
        self.log('val_type_accuracy', type_accuracy, on_epoch=True)
        
        return {'loss': total_loss}
    
    def configure_optimizers(self):
        """Set up optimizer with cosine annealing scheduler."""
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
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step for inference.
        
        Args:
            batch: Tuple of (inputs, frequencies, filter_types)
            batch_idx: Index of the current batch
            
        Returns:
            Tuple of (frequency_pred, type_pred)
        """
        inputs, _, _ = batch
        inputs = inputs.to(self.device).float()
        return self(inputs)


# =============================================================================
# PREDICTOR CLASS
# =============================================================================

class LPFPredictor:
    """
    Main class for training and predicting LPF frequency and filter type.
    
    Prediction Output:
        - frequency: Normalized frequency in [0, 1]
        - filter_type: Encoded filter type index
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self._device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self._device}")
        
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.num_classes = LPFDataset.NUM_CLASSES
        
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
        type_loss_weight: float = 0.5
    ) -> None:
        """
        Train the multi-output prediction model.
        """
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
            num_classes=self.num_classes,
            learning_rate=learning_rate,
            freq_loss_weight=freq_loss_weight,
            type_loss_weight=type_loss_weight
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
        print(f"Filter types: {self.num_classes}")
        print("-" * 60)
        
        trainer.fit(model, datamodule)
        
        print("-" * 60)
        print("Training completed!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
        
        self.model = model.to(self._device)
    
    def predict(self, wav_path: str) -> Tuple[float, int]:
        """
        Predict LPF frequency and filter type for a single audio file.
        
        Returns:
            Tuple of (frequency_normalized, filter_type_index)
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
            freq_pred, type_pred = self.model(features)
        
        frequency = freq_pred.item()
        filter_type = type_pred.argmax(dim=1).item()
        
        return frequency, filter_type
    
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
                'num_classes': self.num_classes,
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self._device)
        
        if self.model is None:
            n_mels = checkpoint.get('n_mels', 128)
            num_classes = checkpoint.get('num_classes', LPFDataset.NUM_CLASSES)
            self.model = LPFMultiOutput(
                input_channels=1,
                n_mels=n_mels,
                num_classes=num_classes
            ).to(self._device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_dataset_statistics(data_dir: str) -> Dict[str, Any]:
    """Compute and display dataset statistics."""
    json_files = sorted(Path(data_dir).glob("*.json"))
    frequencies = []
    filter_types = defaultdict(int)
    
    for jf in json_files:
        try:
            with open(jf) as f:
                config = json.load(f)
            
            freq = float(config["filter_1_freq_hz"])
            frequencies.append(freq)
            
            ft = config.get("filter_1_type", "Unknown")
            filter_types[ft] += 1
        except Exception as e:
            print(f"Warning: Could not process {jf.name}: {e}")
    
    stats = {
        'total_samples': len(json_files),
        'min_frequency': min(frequencies) if frequencies else 0,
        'max_frequency': max(frequencies) if frequencies else 0,
        'mean_frequency': np.mean(frequencies) if frequencies else 0,
        'std_frequency': np.std(frequencies) if frequencies else 0,
        'median_frequency': np.median(frequencies) if frequencies else 0,
        'frequency_range': f"{min(frequencies)} - {max(frequencies)} Hz" if frequencies else "N/A",
        'filter_types': dict(filter_types),
        'num_filter_types': len(filter_types)
    }
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Frequency range: {stats['frequency_range']}")
    print(f"Mean frequency: {stats['mean_frequency']:.1f} Hz")
    print(f"Std deviation: {stats['std_frequency']:.1f} Hz")
    print(f"Median frequency: {stats['median_frequency']:.1f} Hz")
    print(f"\nFilter types distribution ({stats['num_filter_types']} types):")
    for ftype, count in sorted(filter_types.items()):
        print(f"  {ftype}: {count}")
    print("=" * 60 + "\n")
    
    return stats


def predict_with_confidence(
    predictor: LPFPredictor,
    wav_path: str,
    n_samples: int = 10
) -> Dict[str, Any]:
    """
    Make prediction with confidence interval using Monte Carlo dropout.
    
    Returns:
        Dictionary with frequency and filter type predictions
    """
    import soundfile as sf
    
    if predictor.model is None:
        raise ValueError("Model not loaded.")
    
    predictor.model.eval()
    predictor.model.to(predictor._device)
    
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
    mel_normalized = predictor._normalize_mel(mel_db)
    features = torch.FloatTensor(mel_normalized).unsqueeze(0).to(predictor._device)
    
    freq_predictions = []
    type_predictions = []
    
    for _ in range(n_samples):
        with torch.no_grad():
            freq_pred, type_pred = predictor.model(features)
            freq_predictions.append(freq_pred.item())
            type_predictions.append(type_pred.argmax(dim=1).item())
    
    freq_predictions = np.array(freq_predictions)
    type_predictions = np.array(type_predictions)
    
    # Get most common filter type
    unique_types, counts = np.unique(type_predictions, return_counts=True)
    majority_type = unique_types[np.argmax(counts)]
    
    return {
        'frequency': freq_predictions.mean(),
        'frequency_hz': freq_predictions.mean() * 9950 + 50,
        'frequency_std': freq_predictions.std(),
        'frequency_ci_95': [
            (freq_predictions.mean() - 1.96 * freq_predictions.std()) * 9950 + 50,
            (freq_predictions.mean() + 1.96 * freq_predictions.std()) * 9950 + 50
        ],
        'filter_type': majority_type,
        'filter_type_confidence': float(counts[np.argmax(counts)]) / n_samples,
        'n_samples': n_samples
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train or predict LPF frequency and filter type for Serum 2 synthesizer audio',
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
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true',
                           help='Train a new model')
    mode_group.add_argument('--predict', action='store_true',
                           help='Predict LPF for a single audio file')
    mode_group.add_argument('--batch-predict', action='store_true',
                           help='Batch predict LPF for multiple audio files')
    mode_group.add_argument('--stats', action='store_true',
                           help='Display dataset statistics')
    
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing training data (.WAV and .JSON files)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--n-mels', type=int, default=128,
                       help='Number of Mel frequency bins (default: 128)')
    
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model file')
    parser.add_argument('--input-wav', type=str,
                       help='Input audio file for prediction')
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing audio files for batch prediction')
    parser.add_argument('--output-csv', type=str,
                       help='Output CSV file for batch predictions')
    parser.add_argument('--confidence-samples', type=int, default=10,
                        help='Number of samples for confidence estimation (default: 10)')
    
    # Loss weight arguments
    parser.add_argument('--freq-loss-weight', type=float, default=1.0,
                        help='Weight for frequency loss (default: 1.0)')
    parser.add_argument('--type-loss-weight', type=float, default=0.5,
                        help='Weight for filter type loss (default: 0.5)')
    
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
            type_loss_weight=args.type_loss_weight
        )
    
    elif args.predict:
        if not args.input_wav:
            print("Error: --input-wav required for prediction mode")
            sys.exit(1)
        
        frequency, filter_type = predictor.predict(args.input_wav)
        freq_hz = frequency * 9950 + 50
        
        print(f"Predicted LPF frequency (normalized): {frequency:.4f}")
        print(f"LPF frequency in Hz: {freq_hz:.2f} Hz")
        print(f"Predicted filter type: {LPFDataset.FILTER_TYPES[filter_type]}")
    
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
            frequency, filter_type = predictor.predict(str(wav_file))
            freq_hz = frequency * 9950 + 50
            results.append({
                'file': wav_file.name,
                'lpf_frequency_normalized': frequency,
                'lpf_frequency_hz': freq_hz,
                'filter_type_index': filter_type,
                'filter_type_name': LPFDataset.FILTER_TYPES[filter_type]
            })
        
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nBatch predictions saved to {args.output_csv}")
    
    elif args.stats:
        if not args.data_dir:
            print("Error: --data-dir required for statistics mode")
            sys.exit(1)
        
        compute_dataset_statistics(args.data_dir)


if __name__ == '__main__':
    main()
