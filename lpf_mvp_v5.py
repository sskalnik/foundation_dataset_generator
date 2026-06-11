#!/usr/bin/env python3
"""
LPF Frequency & Filter Type Prediction for Serum 2 Synthesizer
==============================================================

Multi-output model predicting:
1. filter_1_freq_hz - LPF cutoff frequency (regression)
2. filter_1_type - Filter type/name (classification)

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

KEY IMPROVEMENTS IN v3:
1. Feature Caching - Precompute mel spectrograms to SSD
2. Hierarchical Caching - SSD → RAM → VRAM pipeline
3. Aggressive Prefetching - 8 workers, 64 prefetch buffer
4. Pinned Memory - Faster CPU→GPU transfers
5. Memory-Mapped Files - Efficient large dataset handling

Author: sskalnik@sskalnik.com
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import sys
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, DeviceStatsMonitor, Timer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.tuner import Tuner

# Audio processing (TODO: replace with something faster)
import librosa
# torchaudio may have issues with newer versions on Windows...
import soundfile as sf

# =============================================================================
# CONFIGURATION
# =============================================================================
# Enable cuDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True
# Set float32 matmul precision for faster training
torch.set_float32_matmul_precision('high')
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
#torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
#torch.backends.cudnn.allow_tf32 = True


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


# ==============================================================================
# COLLATE FUNCTION (Module-level for Windows multiprocessing compatibility)
# ==============================================================================
def v3_collate_fn(batch: List[Tuple[torch.Tensor, Tuple[float, int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for LPF dataset batches.
    Concatenates features and extracts labels in a single pass.
    
    Combines individual samples into batch tensors:
    - Stacks features into a single tensor
    - Stacks normalized frequencies into a tensor
    - Stacks filter type indices into a tensor
    
    Args:
        batch: List of tuples (features, (frequency, filter_type))
        
    Returns:
        Tuple of (features, frequencies, filter_types)
    """
    # Extract and stack features
    features = torch.stack([sample[0].float() for sample in batch])
    # Extract and stack frequencies
    frequencies = torch.stack([torch.tensor(sample[1][0]).float() for sample in batch])
    # Extract and stack filter types
    filter_types = torch.stack([torch.tensor(sample[1][1]).long() for sample in batch])
    return features, frequencies, filter_types
    
    
# ==============================================================================
# Originally this was inside various classes, but that causes issues:
# https://discuss.pytorch.org/t/w-cudaipctypes-cpp-22-producer-process-has-been-terminated-before-all-shared-cuda-tensors-released-see-note-sharing-cuda-tensors/124445/14
# ==============================================================================
def v4_collate_fn(batch: List[Tuple[torch.Tensor, Tuple[float, int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for efficient batch construction.
    Concatenates features and extracts labels in a single pass.
    
    Args:
        batch: List of (features, (frequency, filter_type)) tuples
        
    Returns:
        Tuple of (features_tensor, frequencies_tensor, filter_types_tensor)
    """
    features = torch.stack([b[0] for b in batch])
    frequencies = torch.tensor([b[1][0] for b in batch])
    filter_types = torch.tensor([b[1][1] for b in batch])
    
    return features, frequencies, filter_types


# =============================================================================
# CACHED FEATURE STRUCTURE
# =============================================================================
@dataclass
class CachedFeature:
    """
    Structure for precomputed audio features.
    
    Stores:
        - mel_spectrogram: Precomputed mel spectrogram (n_mels, time)
        - lpf_frequency: Normalized LPF frequency (0-1)
        - filter_type: Encoded filter type index
        - audio_hash: MD5 of audio file for validation
    
    Size estimation per sample:
        - mel_spectrogram: 128 × 93 × 4 bytes ≈ 48 KB
        - lpf_frequency: 4 bytes
        - filter_type: 4 bytes
        - audio_hash: 32 bytes
        Total: ~50 KB per sample
        
    For 100,000 samples: ~5 GB (manageable in RAM)
    
    Attributes:
        mel_spectrogram: 2D numpy array of shape (n_mels, time_frames)
        lpf_frequency: Normalized frequency in [0, 1]
        filter_type: Integer index of filter type
        audio_hash: MD5 hash of audio file for change detection
    """
    mel_spectrogram: np.ndarray
    lpf_frequency: float
    filter_type: int
    audio_hash: str
    
    
class FeatureCache:
    """
    Manages precomputed audio features with multiple storage levels.
    
    Storage Hierarchy:
    ┌─────────────────────────────────────────────────────────────┐
    │ Level 1: VRAM (GPU)        ← Fastest, limited capacity     │
    │   └── Current batch being trained                           │
    │                                                             │
    │ Level 2: RAM (System)      ← Large cache, fast access      │
    │   └── Precomputed features (configurable size)             │
    │                                                             │
    │ Level 3: SSD (Persistent)  ← Largest capacity, slower      │
    │   └── Full dataset cache on disk                           │
    └─────────────────────────────────────────────────────────────┘
    
    Usage Pattern:
    - On first load: Compute features → Save to SSD cache
    - On subsequent runs: Load from SSD → Cache in RAM
    - During training: Prefetch to VRAM asynchronously
    
    Attributes:
        cache_dir: Directory for storing cached features
        ram_cache_size: Maximum number of samples in RAM cache
        preload_to_vram: Whether to preload all features to VRAM
        device: PyTorch device for tensor operations
    """
    def __init__(
        self,
        cache_dir: str = "./feature_cache",
        ram_cache_size: int = 4096,  # Samples to keep in RAM: 8 GB VRAM / 11.5 used when 10_000 ram_cache_size = 0.69, so keep it under 6900
        preload_to_vram: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize feature cache manager.
        
        Args:
            cache_dir: Directory path for caching features on disk
            ram_cache_size: Maximum number of samples to cache in RAM
            preload_to_vram: If True, preloads all features to VRAM on init
            device: PyTorch device (CPU or GPU) for tensor operations
        """
        print("DEBUG: Initializing FeatureCache...")
        self.cache_dir = Path(cache_dir)
        self.ram_cache_size = ram_cache_size
        self.preload_to_vram = preload_to_vram
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # RAM cache with LRU eviction (OrderedDict for LRU tracking)
        self.ram_cache: OrderedDict[int, CachedFeature] = OrderedDict()
        # Precomputed feature files directory
        self.feature_dir = self.cache_dir / 'features'
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: FeatureCache will store .pkl files at {self.feature_dir}")
        # Index mapping sample_idx → feature_file
        self.index_file = self.cache_dir / 'index.pkl'
        self.index: Dict[int, str] = {}
        # Track cached samples for VRAM prefetch
        self.vram_cache: Dict[int, torch.Tensor] = {}
        # Statistics for monitoring
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'compute_count': 0,
            'disk_reads': 0,
            'ram_hits': 0,
            'vram_hits': 0
        }
        
    def get_sample_hash(self, wav_path: Path) -> str:
        """
        Generate MD5 hash for audio file to detect changes.
        
        Reads file in chunks to handle large files efficiently.
        
        Args:
            wav_path: Path to the audio file
            
        Returns:
            Hexadecimal MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(wav_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_or_compute_feature(
        self,
        wav_path: Path,
        json_path: Path,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 8.0,
        fmax: float = 24000.0
    ) -> CachedFeature:
        """
        Get cached feature or compute and cache it.
        
        Strategy:
        1. Check if feature exists on disk with matching hash
        2. If not, compute and save to disk
        3. Load into RAM cache
        4. Evict oldest if cache full
        
        Args:
            wav_path: Path to audio file
            json_path: Path to JSON config
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            fmin: Minimum frequency
            fmax: Maximum frequency
            
        Returns:
            CachedFeature with precomputed values
        """
        audio_hash = self.get_sample_hash(wav_path)
        cache_key = wav_path.stem
        
        # Check if cached feature exists with matching hash
        cache_file = self.feature_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            # Verify hash matches (audio hasn't changed)
            if cached.audio_hash == audio_hash:
                # Feature is valid, add to RAM cache
                self._add_to_ram_cache(cache_key, cached)
                self.stats['cache_hits'] += 1
                self.stats['ram_hits'] += 1
                return cached
        
        # Compute features (not cached or hash mismatch)
        self.stats['compute_count'] += 1
        self.stats['cache_misses'] += 1
        
        # Load audio
        audio_data, sample_rate = sf.read(str(wav_path), dtype='float32')
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax
        )
        # Convert to dB
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Normalize to [0, 1]
        mean = np.mean(mel_db, axis=1, keepdims=True)
        std = np.std(mel_db, axis=1, keepdims=True) + 1e-8
        normalized = (mel_db - mean) / std
        normalized = np.clip(normalized, -50, 50)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        
        # Load JSON config
        with open(json_path) as f:
            config = json.load(f)
        
        # Extract labels
        lpf_frequency = float(config["filter_1_freq_hz"])
        lpf_frequency = np.clip(lpf_frequency, 8.0, 22050.0)
        lpf_frequency = (lpf_frequency - 8.0) / 22042.0  # Normalized
        
        filter_type_str = config.get("filter_1_type", "Unknown")
        if not filter_type_str or filter_type_str == "None":
            filter_type_str = "Unknown"
        
        # Create cached feature
        cached = CachedFeature(
            mel_spectrogram=normalized.astype(np.float32),  # 4 bytes per value
            lpf_frequency=float(lpf_frequency),
            filter_type=LPFDataset.FILTER_TO_INDEX.get(filter_type_str, 0),
            audio_hash=audio_hash
        )
        # Save to disk cache
        print(f"DEBUG: Dumping pickled CachedFeature with cache_key {cache_key}")
        with open(cache_file, 'wb') as f:
            pickle.dump(cached, f)
        # Add to RAM cache
        self._add_to_ram_cache(cache_key, cached)
        return cached
    
    def _add_to_ram_cache(self, key: str, feature: CachedFeature) -> None:
        """
        Add feature to RAM cache with LRU eviction.
        
        Uses OrderedDict to track access order for LRU policy.
        When cache is full, removes least recently used item.
        
        Args:
            key: Cache key (wav file stem)
            feature: CachedFeature to store
        """
        if key in self.ram_cache:
            #print(f"DEBUG: RAM cache already contains CachedFeature with key {key}")
            self.ram_cache.move_to_end(key)
        else:
            #print(f"DEBUG: Adding CachedFeature with key {key} to RAM cache")
            self.ram_cache[key] = feature
            # Evict if cache full
            while len(self.ram_cache) > self.ram_cache_size:
                self.ram_cache.popitem(last=False)
    
    def preload_to_vram(self, dataset: 'CachedLPFDataset') -> None:
        """
        Preload all features to VRAM for fastest access.
        
        Moves cached features from RAM to GPU memory.
        This eliminates data transfer latency during training.
        
        Args:
            dataset: CachedLPFDataset instance
        """
        print(f"Preloading {self.ram_cache_size} of {len(dataset.wav_paths)} samples to VRAM...")
        
        for idx in range(len(dataset.wav_paths)):
            features, _ = dataset[idx]
            # Store tensor directly in VRAM
            self.vram_cache[idx] = features.to(
                self.device, 
                non_blocking=True
            )
        
        print(f"Preload complete. {len(self.vram_cache)} samples in VRAM.")
        print(f"DEBUG: FeatureCache stats: {self.stats}")
        
    def get_ram_cache_stats(self) -> Dict[str, int]:
        """Get RAM cache statistics."""
        return {
            'current_size': len(self.ram_cache),
            'max_size': self.ram_cache_size,
            'utilization': len(self.ram_cache) / self.ram_cache_size
        }
        
    def get_stats(self) -> Dict[str, int]:
        """Get all cache statistics."""
        return self.stats


# =============================================================================
# OPTIMIZED DATASET WITH CACHING
# =============================================================================
class CachedLPFDataset(Dataset):
    """
    Dataset with precomputed features and aggressive caching.
    
    Key Features:
    1. Precomputed features stored on disk
    2. In-memory LRU cache for fast access
    3. Optional VRAM prefetch for reduced latency
    4. Memory-mapped file support for large datasets
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                        CachedLPFDataset                     │
    ├─────────────────────────────────────────────────────────────┤
    │  __getitem__(idx)                                           │
    │      │                                                      │
    │      ▼                                                      │
    │  ┌─────────────────────────────────────────┐                │
    │  │  1. Check VRAM Cache (fastest)          │                │
    │  │  2. Check RAM Cache (fast)              │                │
    │  │  3. Load from SSD (persistent)          │                │
    │  │  4. Compute if not cached (one-time)    │                │
    │  └─────────────────────────────────────────┘                │
    │      │                                                      │
    │      ▼                                                      │
    │  Return (features_tensor, (frequency, filter_type))         │
    └─────────────────────────────────────────────────────────────┘
    
    Performance Expectations:
    - First run: ~5-10 min (precomputation)
    - Subsequent runs: ~10-30 sec (cache load)
    - Training speed: 2-3x faster than original
    """
    def __init__(
        self,
        wav_paths: List[Path],
        feature_cache: FeatureCache,
        transform: Optional[callable] = None,
        augment: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize cached dataset.
        
        Args:
            wav_paths: List of paths to .WAV files
            feature_cache: FeatureCache instance for managing precomputed features
            transform: Optional transform function for data augmentation
            augment: Whether to apply random augmentations
            device: Device for tensor conversion (CPU or GPU)
        """
        self.wav_paths = wav_paths
        self.feature_cache = feature_cache
        self.transform = transform
        self.augment = augment
        self.device = device or torch.device('cpu')
        
        # Preload all features to VRAM if enabled
        if feature_cache.preload_to_vram and device.type == 'cuda':
            self._preload_to_vram()
    
    def _preload_to_vram(self) -> None:
        """Preload all features to VRAM for fastest access."""
        print(f"CachedLPFDataset is preloading {len(self.wav_paths)} samples to VRAM...")
        
        for idx in range(len(self.wav_paths)):
            features, _ = self[idx]
            # Store tensor directly in VRAM
            self.feature_cache.vram_cache[idx] = features.to(self.device, non_blocking=True)
        
        print(f"Preload complete. {len(self.feature_cache.vram_cache)} samples in VRAM.")
    
    def __len__(self) -> int:
        return len(self.wav_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, int]]:
        """
        Get sample with hierarchical caching.
        
        Priority:
        1. VRAM cache (if preloaded)
        2. RAM cache
        3. Disk cache
        4. Compute fresh
        
        Returns:
            Tuple of (features_tensor, (frequency_normalized, filter_type_index))
        """
        wav_path = self.wav_paths[idx]
        json_path = wav_path.with_stem(f"{wav_path.stem}_params").with_suffix('.json')
        
        # Check VRAM cache first (fastest)
        if idx in self.feature_cache.vram_cache:
            features = self.feature_cache.vram_cache[idx]
            # Recompute labels since they're small
            config = json.load(open(json_path))
            lpf_frequency = self._extract_lpf_frequency(config)
            filter_type = self._extract_filter_type(config)
            return features, (lpf_frequency, filter_type)
        
        # Get from feature cache
        cached = self.feature_cache.get_or_compute_feature(wav_path, json_path)
        
        # Convert to tensor
        features = torch.FloatTensor(cached.mel_spectrogram).unsqueeze(0)
        
        # Apply transform/augmentation if enabled
        if self.transform:
            features = self.transform(features)
        
        if self.augment:
            features = self._augment(features)
        
        # Move to device
        features = features.to(self.device)
        
        return features, (cached.lpf_frequency, cached.filter_type)
    
    def _augment(self, features: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to features."""
        # Random time shift (simulate different audio positions)
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            features = torch.roll(features, shift, dims=-1)
        
        # Random gain variation
        if random.random() < 0.5:
            gain = random.uniform(0.9, 1.1)
            features = features * gain
        
        # Random noise
        if random.random() < 0.3:
            noise = torch.randn_like(features) * 0.02
            features = features + noise
        
        return features
    
    def _extract_lpf_frequency(self, config: dict) -> float:
        """Extract and normalize LPF frequency."""
        freq_hz = float(config["filter_1_freq_hz"])
        freq_clamped = np.clip(freq_hz, 8.0, 22050.0)
        normalized = (freq_clamped - 8.0) / 22042.0
        return float(normalized)
    
    def _extract_filter_type(self, config: dict) -> int:
        """Extract filter type index."""
        filter_type_str = config.get("filter_1_type", "Unknown")
        if not filter_type_str or filter_type_str == "None":
            filter_type_str = "Unknown"
        return LPFDataset.FILTER_TO_INDEX.get(filter_type_str, 0)


# =============================================================================
# OPTIMIZED DATA MODULE
# =============================================================================
class OptimizedLPFDataModule(LightningDataModule):
    """
    Optimized DataModule with aggressive prefetching and caching.
    
    Key Optimizations:
    1. Multiple workers (8+) for parallel I/O
    2. Large prefetch buffer (64) for GPU saturation
    3. Pin memory for faster CPU→GPU transfer
    4. Persistent workers to avoid restart overhead
    5. Custom collate function for efficient batching
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    DataLoader Pipeline                      │
    ├─────────────────────────────────────────────────────────────┤
    │  Worker 1 ──┐                                               │
    │  Worker 2 ──┼──→ Batch Queue ──→ Prefetch ──→ GPU           │
    │  Worker 3 ──┼──→ Batch Queue ──→ Prefetch ──→ GPU           │
    │  Worker 4 ──┘                                               │
    │                                                             │
    │  num_workers=8: 8 parallel processes                        │
    │  prefetch_factor=64: 64 batches per worker in queue         │
    │  pin_memory=True: Direct CPU→GPU transfer                   │
    │  persistent_workers=True: Workers stay alive                │
    └─────────────────────────────────────────────────────────────┘
    
    Expected Throughput:
    - Before: ~50 batches/sec
    - After: ~150-200 batches/sec (3-4x improvement)
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 1,
        validation_split: float = 0.1,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        cache_dir: str = "./feature_cache",
        ram_cache_size: int = 4096,
        prefetch_factor: int = 128,  # Increased from 32
        pin_memory: bool = False,    # Disable because data already pre-cached to VRAM
        persistent_workers: bool = True,
        preload_to_vram: bool = False,
        augment: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize optimized data module.
        
        Args:
            data_dir: Directory containing .WAV and .JSON files
            batch_size: Batch size for training (default: 128)
            num_workers: Number of dataloader workers (default: 1)
            validation_split: Fraction for validation set
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            cache_dir: Directory for feature cache (default: "./feature_cache")
            ram_cache_size: Number of samples to keep in RAM cache (default: 4096)
            prefetch_factor: Number of batches to prefetch per worker (default: 128)
            pin_memory: Enable pinned memory for faster transfers (default: True)
            persistent_workers: Keep workers alive between epochs (default: True)
            preload_to_vram: Preload features to VRAM on init (default: True)
            augment: Enable data augmentation (default: True)
            device: PyTorch device for tensors
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cache_dir = cache_dir
        self.ram_cache_size = ram_cache_size
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.preload_to_vram = preload_to_vram
        self.augment = augment
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataset: Optional[CachedLPFDataset] = None
        self.val_dataset: Optional[CachedLPFDataset] = None
        self.wav_files: List[Path] = []
        
        # Initialize feature cache
        print(f"DEBUG: OptimizedLPFDataModule is initializing FeatureCache with ram_cache_size = {ram_cache_size}")
        self.feature_cache = FeatureCache(
            cache_dir=cache_dir,
            ram_cache_size=ram_cache_size,
            preload_to_vram=preload_to_vram,
            device=self.device
        )
        
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Initialize datasets with precomputed features.
        
        First run:
        1. Discover all .WAV files
        2. Compute features for all samples (cached to disk)
        3. Split into train/validation
        
        Subsequent runs:
        1. Load cached features from disk
        2. Build RAM cache
        3. Split datasets
        
        Args:
            stage: Training stage (fit, validate, test, predict)
        """
        # Discover all .WAV files
        self.wav_files = sorted(list(self.data_dir.rglob("*.wav")))
        
        if not self.wav_files:
            raise ValueError(f"No .wav files found in {self.data_dir}")
        
        print(f"Found {len(self.wav_files)} audio files")
        
        # Create full dataset with caching
        print('DEBUG: OptimizedLPFDataModule is creating full_dataset, which is a CachedLPFDataset based on OptimizedLPFDataModule.feature_cache')
        full_dataset = CachedLPFDataset(
            wav_paths=self.wav_files,
            feature_cache=self.feature_cache,
            augment=self.augment,
            device=self.device
        )
        
        # Split into train/validation sets
        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size
        
        print(f"Split: {train_size} training, {val_size} validation")
        print(f"Cache stats: {self.feature_cache.get_ram_cache_stats()}")
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(667)
        )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader with optimized settings."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            multiprocessing_context='spawn',  # Better for Windows
            collate_fn=v4_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader with optimized settings."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            multiprocessing_context='spawn',
            collate_fn=v4_collate_fn
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            'ram_cache': self.feature_cache.get_ram_cache_stats(),
            'vram_cache': len(self.feature_cache.vram_cache),
            'disk_cache': len(list(self.feature_cache.feature_dir.glob('*.pkl')))
        }


# =============================================================================
# ORIGINAL DATASET (for backward compatibility)
# =============================================================================
class LPFDataset(Dataset):
    """
    Dataset for loading and preprocessing audio samples with LPF frequency and filter type.
    
    Expected format:
        - .WAV files: 1-second C3 notes at 48kHz, 32-bit float
        - .JSON files: Contains filter frequency configuration
    
    Output format:
        - frequency: normalized LPF frequency in [0, 1]
        - filter_type: encoded filter type class index
    
    The dataset handles:
        - Audio loading via soundfile (preserves 32-bit float)
        - Mel spectrogram computation
        - Normalization for training stability
        
    Target Normalization:
        LPF frequencies are normalized to [0, 1] range using:
            normalized = (frequency - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
        where MIN_FREQ = 8 Hz and MAX_FREQ = 22050 Hz
        
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
    
    # Class-level constants for target normalization
    TARGET_MIN: float = 8.0      # Minimum reasonable LPF frequency (Hz)
    TARGET_MAX: float = 22050.0   # Maximum reasonable LPF frequency (Hz)
    TARGET_RANGE: float = TARGET_MAX - TARGET_MIN
    
    #serum2.filter_1_type.valid_values
    FILTER_TYPES = [
        'MG Low 6', 'MG Low 12', 'MG Low 18', 'MG Low 24',
        'Low 6', 'Low 12', 'Low 18', 'Low 24',
        'High 6', 'High 12', 'High 18', 'High 24',
        'Band 12', 'Band 24',
        'Peak 12', 'Peak 24',
        'Notch 12', 'Notch 24',
        'LH 6', 'LH 12', 'LB 12', 'LP 12', 'LN 12',
        'HB 12', 'HP 12', 'HN 12',
        'BP 12', 'BN 12', 'PP 12', 'PN 12', 'NN 12',
        'L/B/H 12', 'L/B/H 24', 'L/P/H 12', 'L/P/H 24', 'L/N/H 12', 'L/N/H 24', 'B/P/N 12', 'B/P/N 24',
        'Cmb +', 'Cmb -', 'Cmb L6+', 'Cmb L6-', 'Cmb H6+', 'Cmb H6-', 'Cmb HL6+', 'Cmb HL6-',
        'Flg +', 'Flg -', 'Flg L6+', 'Flg L6-', 'Flg H6+', 'Flg H6-', 'Flg HL6+', 'Flg HL6-',
        'Phs 12+', 'Phs 12-', 'Phs 24+', 'Phs 24-', 'Phs 36+', 'Phs 36-', 'Phs 48+', 'Phs 48-',
        'Phs 48L6+', 'Phs 48L6-', 'Phs 48H6+', 'Phs 48H6-', 'Phs 48HL6+', 'Phs 48HL6-',
        'FPhs 12HL6+', 'FPhs 12HL6-',
        'Low EQ 6', 'Low EQ 12', 'Band EQ 12', 'High EQ 6', 'High EQ 12',
        'Ring Mod', 'Ring Modx2', 'SampHold', 'SampHold-', 'Combs', 'Allpasses', 'Reverb',
        'French LP', 'German LP', 'Add Bass',
        'Formant-I', 'Formant-II', 'Formant-III',
        'Bandreject', 'Dist.Comb 1 LP', 'Dist.Comb 1 BP', 'Dist.Comb 2 LP', 'Dist.Comb 2 BP',
        'Scream LP', 'Scream BP', 'Wsp', 'DJ Mixer', 'Diffusor',
        'MG Ladder', 'Acid Ladder', 'EMS Ladder', 'MG Dirty',
        'PZ SVF', 'Comb 2', 'Exp MM', 'Exp BPF', 'K35'
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
        fmin: float = 8.0,
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
        self.wav_paths = wav_paths
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self) -> int:
        return len(self.wav_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[float, int]]:
        """
        Get a single sample with frequency and filter type.
        
        Returns:
            Tuple of (features_tensor, (frequency_normalized, filter_type_index))
            where lpf_frequency is normalized to [0, 1] range
        """
        wav_path = self.wav_paths[idx]
        json_path = wav_path.with_stem(f"{wav_path.stem}_params").with_suffix('.json')
        
        # Load audio at native sample rate (48kHz) using float32
        audio_data, sample_rate = sf.read(str(wav_path), dtype='float32')
        
        # Handle stereo: convert to mono by averaging channels
        if len(audio_data.shape) > 1:
            # Stereo audio - average channels to get mono
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
        normalized = self._normalize_mel(mel_db)
        
        # Convert to torch tensor with shape (channels, height, width)
        features_tensor = torch.FloatTensor(normalized).unsqueeze(0)
        
        # Load JSON configuration and extract LPF frequency
        config = json.load(open(json_path))
        lpf_frequency_normalized = self._extract_lpf_frequency(config)
        filter_type_index = self._extract_filter_type(config)
        
        return features_tensor, (lpf_frequency_normalized, filter_type_index)
    
    def _normalize_mel(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize Mel spectrogram values.
        
        Standardizes to zero mean and unit variance per frequency band,
        then clips and rescales to [0, 1] range.
        
        Args:
            mel_spectrogram: 2D numpy array of Mel spectrogram values
            
        Returns:
            Normalized Mel spectrogram with values in [0, 1]
        """
        # Per-band normalization
        mean = np.mean(mel_spectrogram, axis=1, keepdims=True)
        std = np.std(mel_spectrogram, axis=1, keepdims=True) + 1e-8
        normalized = (mel_spectrogram - mean) / std
        # Clip to reasonable range and rescale
        normalized = np.clip(normalized, -50, 50)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)    
        return normalized
    
    def _extract_lpf_frequency(self, config: dict) -> float:
        """
        Extract LPF frequency from configuration dictionary and normalize.
        
        Serum 2 JSON structure uses "filter_1_freq_hz" for the low-pass 
        filter cutoff frequency in Hertz. This is the parameter we want to predict.
        
        The frequency is normalized to [0, 1] range using:
            normalized = (frequency - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
            
        where TARGET_MIN = 8 Hz and TARGET_MAX = 22050 Hz
        
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
        # Serum 2 uses "filter_1_freq_hz" for the filter cutoff frequency
        freq_hz = float(config["filter_1_freq_hz"])
        # Clamp to valid range
        freq_clamped = np.clip(freq_hz, self.TARGET_MIN, self.TARGET_MAX)
        # Normalize to [0, 1]
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
        """
        Convert normalized frequency back to Hz.
        
        Args:
            normalized: Normalized frequency in [0, 1] range
            
        Returns:
            Frequency in Hz
        """
        return normalized * self.TARGET_RANGE + self.TARGET_MIN
    
    @staticmethod
    def compute_statistics(data_dir: str) -> Dict[str, Any]:
        """
        Compute dataset statistics for reporting.
        
        Analyzes all JSON files in the data directory and computes
        statistics about the LPF frequency distribution.
        
        Args:
            data_dir: Path to directory containing .JSON files
            
        Returns:
            Dictionary with frequency distribution, counts, etc.
        """
        json_files = sorted(Path(data_dir).rglob("*.json"))
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
            'num_filter_types': len(filter_types),
            'files_processed': filter_counts
        }
    

# =============================================================================
# ORIGINAL DATA MODULE (for backward compatibility)
# =============================================================================
class LPFDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule managing the LPF dataset.
    
    Handles data splitting, batching, and DataLoader creation.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 1,
        validation_split: float = 0.1,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        prefetch_factor: int = 128
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.prefetch_factor = prefetch_factor
        
        self.train_dataset: Optional[LPFDataset] = None
        self.val_dataset: Optional[LPFDataset] = None
        self.wav_files: List[Path] = []
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets before training."""
        # Discover all .WAV files
        self.wav_files = sorted(list(self.data_dir.rglob("*.wav")))
        
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
        print(f"Number of filter types: {LPFDataset.NUM_CLASSES}")
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(667)
        )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader with CUDA support."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False, # Disable because data already pre-cached to VRAM
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,  # Prefetch batches
            collate_fn=lpf_collate_fn,  # Use module-level function
            multiprocessing_context='spawn'  # Better for Windows
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False, # Disable because data already pre-cached to VRAM
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,  # Prefetch batches
            collate_fn=lpf_collate_fn,  # Use module-level function
            multiprocessing_context='spawn'  # Better for Windows
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
    
    Architecture designed for:
    - Input: Mel spectrogram (1 × 128 × ~93)
    - Output: Single float value (normalized LPF frequency in [0, 1])
    """
    
    TARGET_MIN: float = 8.0
    TARGET_MAX: float = 22050.0
    TARGET_RANGE: float = TARGET_MAX - TARGET_MIN
    
    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        num_classes: int = LPFDataset.NUM_CLASSES,
        learning_rate: float = 0.0005,
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
        self.flattened_size: Optional[int] = None
        
        # Convolutional feature extractor (shared)
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
        
        # Dynamic head initialization - will be set in forward
        self.freq_head: Optional[nn.Sequential] = None
        self.type_head: Optional[nn.Sequential] = None
        
    def _initialize_heads(self, input_features: int) -> None:
        """Initialize frequency and type heads based on actual input features."""
        self.freq_head = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output in [0, 1]
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
        # Conv blocks are already on the correct device
        x = self.conv_blocks(x)
        
        # Dynamic flattening - capture actual dimensions
        actual_flattened_size = x.size(1) * x.size(2) * x.size(3)
        
        # Initialize heads if not already done
        if self.freq_head is None:
            self._initialize_heads(actual_flattened_size)
            
        # Move head layers to correct device
        for layer in self.freq_head:
            layer.to(self.device)
        for layer in self.type_head:
            layer.to(self.device)
        
        # Reshape to match actual flattened size
        x = x.view(x.size(0), actual_flattened_size)
        
        # Pass through heads
        freq_pred = self.freq_head(x)
        type_pred = self.type_head(x)
        
        return freq_pred, type_pred
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Training step with gradient clipping and combined loss.
        
        Args:
            batch: Tuple of (inputs, frequencies, filter_types)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and logged metrics
        """
        inputs, frequencies, filter_types = batch
        
        # Move to device and ensure correct dtypes
        inputs = inputs.float()
        frequencies = frequencies.float().unsqueeze(1)
        filter_types = filter_types.long()
        
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
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('train_freq_loss', freq_loss, on_epoch=True, prog_bar=True)
        self.log('train_type_loss', type_loss, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('train_type_accuracy', type_accuracy, on_epoch=True, prog_bar=True)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=True, prog_bar=True)
        
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
        
        # Explicitly move to model's device
        inputs = inputs.float()
        frequencies = frequencies.float().unsqueeze(1)
        filter_types = filter_types.long()
        
        freq_pred, type_pred = self(inputs)
        
        freq_loss = nn.MSELoss()(freq_pred, frequencies)
        type_loss = nn.CrossEntropyLoss()(type_pred, filter_types.squeeze())
        total_loss = self.freq_loss_weight * freq_loss + self.type_loss_weight * type_loss
        
        mae = nn.L1Loss()(freq_pred, frequencies)
        rmse = torch.sqrt(freq_loss)
        type_accuracy = (type_pred.argmax(dim=1) == filter_types.squeeze()).float().mean()
        
        # Log validation metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_freq_loss', freq_loss, on_epoch=True, prog_bar=True)
        self.log('val_type_loss', type_loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('val_type_accuracy', type_accuracy, on_epoch=True, prog_bar=True)
        
        return {'loss': total_loss}
    
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
        
        scheduler_cawr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # Warmup epochs
            T_mult=2,    # Expand cycle length
            eta_min=1e-6
        )
        
        scheduler_rlop = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,      # Halve LR when plateau detected
            patience=10,     # Wait 10 epochs before reducing
            min_lr=1e-6
        )
        
        scheduler = scheduler_cawr
        
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
    
    Provides a high-level interface for:
        - Training models with PyTorch Lightning
        - Making predictions on audio files
        - Managing model checkpoints
        
    Prediction Output:
        - frequency: Normalized frequency in [0, 1]
        - filter_type: Encoded filter type index
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
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.num_classes = LPFDataset.NUM_CLASSES
        
        # Model instance
        self.model: Optional[LPFMultiOutput] = None
        
        # Load pretrained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(
        self,
        data_dir: str,
        epochs: int = 1000,
        batch_size: int = 128,
        prefetch_factor: int = 128,
        num_workers: int = 1,
        ram_cache_size: int = 4096,
        learning_rate: float = 0.0005,
        validation_split: float = 0.1,
        output_dir: str = "./models",
        n_mels: int = 128,
        freq_loss_weight: float = 1.0,
        type_loss_weight: float = 0.5,
        use_optimized: bool = True  # Use OptimizedLPFDataModule if True
    ) -> None:
        """
        Train the multi-output prediction model.
        
        Args:
            data_dir: Directory containing .WAV and .JSON files
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction for validation set
            output_dir: Directory to save checkpoints
            n_mels: Number of Mel frequency bins
            use_optimized: If True, uses OptimizedLPFDataModule with caching
        """
        if use_optimized:
            # Use optimized datamodule with caching
            print(f"DEBUG: predictor.train() is creating datamodule = OptimizedLPFDataModule() with ram_cache_size = {ram_cache_size}")
            datamodule = OptimizedLPFDataModule(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                validation_split=validation_split,
                n_mels=n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                prefetch_factor=prefetch_factor,
                cache_dir="./feature_cache",
                ram_cache_size=ram_cache_size,
                preload_to_vram=False,
                augment=False
            )
        else:
            # Use original datamodule for backward compatibility
            datamodule = LPFDataModule(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                validation_split=validation_split,
                n_mels=n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                prefetch_factor=prefetch_factor
            )
        
        # Setup callbacks with timestamped names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path =f"best_{timestamp}" + '_epoch{epoch:04d}_step{step}_val_loss{val_loss:.4g}_val_freq_loss{val_freq_loss:.4g}_val_type_loss{val_type_loss:.4g}'
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=checkpoint_path,
            save_top_k=5,
            monitor='val_loss',
            mode='min',
            verbose=True,
            auto_insert_metric_name=False
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min',
            verbose=True
        )
        
        rich_model_summary = RichModelSummary(max_depth=-1)
        device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
        time_stats_monitor = Timer(duration=None, verbose=True)
        
        # Setup logger
        logger = TensorBoardLogger('tb_logs', name=f'lpf_prediction_{timestamp}')
        
        # Create trainer with mixed precision support
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision='16-mixed',  # Mixed precision for faster training
            callbacks=[checkpoint_callback, early_stop_callback, device_stats_monitor, time_stats_monitor, rich_model_summary],
            logger=logger,
            log_every_n_steps=50,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            profiler="simple"
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Filter types: {self.num_classes}")
        print(f"Batch size: {batch_size}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Optimized mode (pre-transform dataset and load into system RAM (maybe also into VRAM)): {use_optimized}")
        print("-" * 60)
        
        # Setup model
        print('DEBUG: predictor.train() is creating model = LPFMultiOutput()')
        model = LPFMultiOutput(
            input_channels=1,
            n_mels=n_mels,
            num_classes=self.num_classes,
            learning_rate=learning_rate,
            freq_loss_weight=freq_loss_weight,
            type_loss_weight=type_loss_weight
        )
        
        #print('Tuning the batch size...')
        #tuner = Tuner(trainer)
        # None = default | "power" = powers of 2 til OOM | "binsearch" = "power" til OOM, then fine-tune
        # Auto-scale batch size by growing it exponentially (default)
        #tuner.scale_batch_size(model, datamodule, mode="power")
        # Auto-scale batch size with binary search
        #tuner.scale_batch_size(model, mode="binsearch")
        #print('Tuning the learning rate...')
        # Run learning rate finder
        #lr_finder = tuner.lr_find(model)
        # Results can be found in
        #print(f"lr_finder.results = {lr_finder.results}")
        # Plot with
        #fig = lr_finder.plot(suggest=True)
        #fig.show()
        # Pick point based on plot, or get suggestion
        #new_lr = lr_finder.suggestion()
        # update hparams of the model
        #model.learning_rate = new_lr
        # Fit as normal with new batch size
        print('DEBUG: Calling trainer.fit(model, datamodule)')
        trainer.fit(model, datamodule)
        
        print("-" * 60)
        print("Training completed!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Best val loss: {checkpoint_callback.best_model_score:.6f}")
        
        self.model = model
        
        save_model_path = os.path.join(output_dir, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        print(f"Saving best model to {save_model_path}")
        self.save_model(save_model_path)
    
    def predict(self, wav_path: str) -> Tuple[float, int]:
        """
        Predict LPF frequency and filter type for a single audio file.
        
        Args:
            wav_path: Path to the .WAV file
            
        Returns:
            Tuple of (frequency_normalized, filter_type_index)
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
            fmin=8,
            fmax=24000
        )
        
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_normalized = self._normalize_mel(mel_db)
        
        # Convert to tensor and predict
        features = torch.FloatTensor(mel_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            freq_pred, type_pred = self.model(features)
        
        frequency = freq_pred.item()
        filter_type = type_pred.argmax(dim=1).item()
        
         # DEBUG: Print diagnostic information
        print(f"\n--- Prediction Debug ---")
        print(f"Input file: {wav_path}")
        print(f"Freq pred (normalized): {frequency:.4f}")
        print(f"Freq pred (Hz): {frequency * 22042 + 8:.2f}")
        print(f"Filter type: {LPFDataset.FILTER_TYPES[filter_type]}")
        print(f"Model freq_head sigmoid output range: [{freq_pred.min():.4f}, {freq_pred.max():.4f}]")
        print(f"------------------------\n")
        
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
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            n_mels = checkpoint.get('n_mels', 128)
            num_classes = checkpoint.get('num_classes', LPFDataset.NUM_CLASSES)
            self.model = LPFMultiOutput(
                input_channels=1,
                n_mels=n_mels,
                num_classes=num_classes
            ).to(self.device)
        
        print(f"DEBUG: Model n_mels = {n_mels}")
        print(f"DEBUG: Model num_classes = {num_classes}")

        print("DEBUG: self.model.load_state_dict(checkpoint['model_state_dict'])")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #print("DEBUG: self.model.load_state_dict(checkpoint['state_dict'])")
        #self.model.load_state_dict(checkpoint['state_dict'])
        print("DEBUG: test prediction with random tensor values:")
        test_input = torch.randn(1, 1, 128, 93).to(self.device)
        test_output = self.model(test_input)
        print(f"Test prediction: {test_output[0].item():.4f}")
        
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
        if info.samplerate != 48000:
            print(f"Warning: {path.name} has sample rate {info.samplerate}, "
                  f"expected 48000")
        # Check duration (should be ~1 second)
        if abs(info.duration - 1.0) > 0.2:
            print(f"Warning: {path.name} duration is {info.duration:.2f}s, "
                  f"expected ~1s")
        return True
    except Exception as e:
        print(f"Error validating {path.name}: {e}")
        return False
        
def compute_dataset_statistics(data_dir: str) -> Dict[str, Any]:
    """
    Compute and display dataset statistics.
    
    Args:
        data_dir: Path to directory containing .JSON files
        
    Returns:
        Dictionary with statistics
    """
    json_files = sorted(Path(data_dir).rglob("*.json"))
    frequencies = []
    filter_types = defaultdict(int)
    
    for jf in json_files:
        try:
            with open(jf) as f:
                config = json.load(f)    
            freq = float(config["filter_1_freq_hz"])
            frequencies.append(freq)
            if "filter_1_type" in config:
                filter_types[config["filter_1_type"]] += 1
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
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Frequency range: {stats['frequency_range']}")
    print(f"Mean frequency: {stats['mean_frequency']:.2f} Hz")
    print(f"Std deviation: {stats['std_frequency']:.2f} Hz")
    print(f"Median frequency: {stats['median_frequency']:.2f} Hz")
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
    
    Runs multiple forward passes and computes statistics on the results.
    This provides an estimate of model uncertainty.
    
    Args:
        predictor: LPFPredictor instance
        wav_path: Path to audio file
        n_samples: Number of forward passes for uncertainty estimation
        
    Returns:
        Dictionary with frequency and filter type predictions
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
        fmin=8,
        fmax=24000
    )
    
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_normalized = predictor._normalize_mel(mel_db)
    features = torch.FloatTensor(mel_normalized).unsqueeze(0).to(predictor.device)
    
    # Multiple predictions with dropout enabled (Monte Carlo)
    predictor.model.eval()
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
        'frequency_hz': freq_predictions.mean() * 22042 + 8,
        'frequency_std': freq_predictions.std(),
        'frequency_ci_95': [
            (freq_predictions.mean() - 1.96 * freq_predictions.std()) * 22042 + 8,
            (freq_predictions.mean() + 1.96 * freq_predictions.std()) * 22042 + 8
        ],
        'filter_type': majority_type,
        'filter_type_confidence': float(counts[np.argmax(counts)]) / n_samples,
        'n_samples': n_samples
    }

def print_model_summary(model: LPFMultiOutput) -> None:
    """
    Print detailed model architecture and parameter counts.
    
    Args:
        model: LPFMultiOutput instance to summarize
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
    print(f"\nParameter breakdown:")
    print(f"  Convolutional layers: {conv_params:,}")
    print("=" * 60 + "\n")


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
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--prefetch-factor', type=int, default=128,
                       help='Integer number of samples to pre-fetch per dataloader worker (default: 128)')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Integer number of workers for data loading (default: 1)')
    parser.add_argument('--ram-cache-size', type=int, default=4096,
                        help='Number of samples to pre-load into RAM cache (default: 4096)')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--n-mels', type=int, default=128,
                       help='Number of Mel frequency bins (default: 128)')
    
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
            prefetch_factor=args.prefetch_factor,
            num_workers=args.num_workers,
            ram_cache_size=args.ram_cache_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            n_mels=args.n_mels,
            freq_loss_weight=args.freq_loss_weight,
            type_loss_weight=args.type_loss_weight,
            use_optimized=True  # Use optimized datamodule by default
        )
        save_path = os.path.join(args.output_dir, f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        predictor.save_model(save_path)
    elif args.predict:
        if not args.input_wav:
            print("Error: --input-wav required for prediction mode")
            sys.exit(1)
        
        frequency, filter_type = predictor.predict(args.input_wav)
        print(f"Predicted LPF frequency (normalized): {frequency:.4f}")
        # Convert to Hz for display
        freq_hz = frequency * 22042 + 8
        print(f"LPF frequency in Hz: {freq_hz:.2f} Hz")
        print(f"Predicted filter type: {LPFDataset.FILTER_TYPES[filter_type]}")
    elif args.batch_predict:
        if not args.input_dir or not args.output_csv:
            print("Error: --input-dir and --output-csv required for batch prediction")
            sys.exit(1)
        input_path = Path(args.input_dir)
        wav_files = list(input_path.rglob('*.wav'))
        results = []
        for wav_file in tqdm(wav_files, desc="Processing files"):
            frequency, filter_type = predictor.predict(str(wav_file))
            freq_hz = frequency * 22042 + 8
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
