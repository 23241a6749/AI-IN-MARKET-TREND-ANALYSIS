"""
Data augmentation techniques for time-series data.
Helps create more training samples from limited data.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class TimeSeriesAugmentation:
    """Augmentation techniques for time-series sequences."""
    
    @staticmethod
    def add_noise(sequence: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add small random noise to sequence.
        
        Args:
            sequence: Input sequence [seq_len, features]
            noise_level: Standard deviation of noise (as fraction of data std)
        
        Returns:
            Augmented sequence
        """
        noise_std = sequence.std() * noise_level
        noise = np.random.normal(0, noise_std, sequence.shape)
        return sequence + noise
    
    @staticmethod
    def time_warp(sequence: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Apply time warping (stretching/compressing time axis).
        
        Args:
            sequence: Input sequence [seq_len, features]
            sigma: Warping strength
        
        Returns:
            Warped sequence
        """
        seq_len = sequence.shape[0]
        
        # Generate random warping
        warp = np.cumsum(1 + np.random.normal(0, sigma, seq_len))
        warp = (warp - warp.min()) / (warp.max() - warp.min()) * (seq_len - 1)
        warp = warp.astype(int)
        warp = np.clip(warp, 0, seq_len - 1)
        
        # Apply warping
        warped = sequence[warp]
        return warped
    
    @staticmethod
    def magnitude_scale(sequence: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Scale magnitude of sequence.
        
        Args:
            sequence: Input sequence [seq_len, features]
            sigma: Scaling factor standard deviation
        
        Returns:
            Scaled sequence
        """
        scaling_factor = 1 + np.random.normal(0, sigma)
        return sequence * scaling_factor
    
    @staticmethod
    def augment_sequences(X: np.ndarray, y: np.ndarray, 
                         augmentation_factor: int = 2,
                         methods: list = ['noise', 'magnitude']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment sequences to increase dataset size.
        
        Args:
            X: Input sequences [n_samples, seq_len, features]
            y: Targets [n_samples]
            augmentation_factor: How many augmented samples per original
            methods: List of augmentation methods to use
        
        Returns:
            Augmented X and y
        """
        augmented_X = []
        augmented_y = []
        
        # Keep original data
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Create augmented versions
        for _ in range(augmentation_factor):
            aug_X = []
            aug_y = []
            
            for i in range(len(X)):
                seq = X[i].copy()
                
                # Apply random augmentation method
                method = np.random.choice(methods)
                
                if method == 'noise':
                    seq = TimeSeriesAugmentation.add_noise(seq, noise_level=0.02)
                elif method == 'magnitude':
                    seq = TimeSeriesAugmentation.magnitude_scale(seq, sigma=0.1)
                elif method == 'time_warp':
                    seq = TimeSeriesAugmentation.time_warp(seq, sigma=0.2)
                
                aug_X.append(seq)
                aug_y.append(y[i])
            
            augmented_X.append(np.array(aug_X))
            augmented_y.append(np.array(aug_y))
        
        # Concatenate all
        X_aug = np.concatenate(augmented_X, axis=0)
        y_aug = np.concatenate(augmented_y, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(X_aug))
        X_aug = X_aug[indices]
        y_aug = y_aug[indices]
        
        return X_aug, y_aug

