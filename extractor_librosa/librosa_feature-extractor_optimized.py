#!/usr/bin/env python3
"""
Enhanced Librosa Feature Extractor for Album Analysis

This script scans a folder for audio files and extracts ALL possible librosa features
plus additional album-specific features for music analysis and arrangement.

Usage: python enhanced_librosa_feature_extractor.py input_folder [output_folder]

Requirements:
    pip install librosa numpy scipy

Author: Enhanced for comprehensive audio feature extraction and album arrangement
"""

import os
import sys
import argparse
import warnings
import numpy as np
import librosa
import librosa.feature
import librosa.onset
import librosa.beat
import librosa.decompose
import librosa.effects
import librosa.segment
import librosa.sequence
import librosa.util
import librosa.filters
from pathlib import Path
import traceback
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EnhancedLibrosaFeatureExtractor:
    def __init__(self, sr=44100, hop_length=128, n_fft=2048):
        """
        Initialize with parameters optimized for album arrangement analysis
        
        Parameter Selection Rationale:
        - sr=44100: Standard audio fidelity, captures full musical frequency range
        - hop_length=128: High temporal resolution for precise beat/rhythm/transition analysis  
        - n_fft=2048: Good frequency balance without excessive memory usage
        
        Alternative Parameter Sets by Use Case:
        ┌─────────────────────────────┬──────┬─────────────┬──────┐
        │ Use Case                    │ sr   │ hop_length  │ n_fft│
        ├─────────────────────────────┼──────┼─────────────┼──────┤
        │ Speech / Podcasts           │16000 │ 256         │ 1024 │
        │ Modern Pop / EDM            │44100 │ 256 or 512  │ 2048 │
        │ Classical / Jazz / Wide Dyn │48000 │ 512         │ 4096 │
        │ Real-time Analysis          │22050 │ 512         │ 1024 │
        │ Spectrogram-heavy Visuals   │44100 │ 128         │ 2048+│
        └─────────────────────────────┴──────┴─────────────┴──────┘
        
        Current settings prioritize temporal precision for album sequencing:
        - Detailed energy trajectory analysis for smooth transitions
        - Precise beat-to-beat timing for rhythm matching
        - High-resolution onset detection for natural cut points
        
        To modify for other use cases, change these parameters in the constructor call.
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = 128
        self.n_mfcc = 13
        self.n_chroma = 12
        
    def extract_all_features(self, audio_path):
        """Extract all possible librosa features plus album-specific features from an audio file"""
        print(f"Processing: {audio_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            features = {}
            
            # Store basic audio info
            # NOTE: Raw audio storage removed for file size optimization (~42MB per track)
            # To re-enable for other applications, uncomment this line:
            # features['audio_raw'] = y
            features['sample_rate'] = sr
            features['duration'] = librosa.get_duration(y=y, sr=sr)
            
            # === CORE IO AND DSP ===
            print("  Extracting core features...")
            
            # Basic audio processing
            features['audio_mono'] = librosa.to_mono(y.reshape(1, -1))
            
            # Time-domain processing
            features['autocorrelation'] = librosa.autocorrelate(y)
            try:
                features['lpc'] = librosa.lpc(y, order=16)
            except:
                features['lpc'] = None
            features['zero_crossings'] = librosa.zero_crossings(y)
            
            # Signal generation (for reference/testing)
            click_track = librosa.clicks(times=np.arange(0, len(y)/sr, 1), sr=sr, length=len(y))
            features['click_track'] = click_track
            
            # Multi-resolution spectral representations
            stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
            features['stft_real'] = np.real(stft)
            features['stft_imag'] = np.imag(stft)
            features['stft_magnitude'] = np.abs(stft)
            features['stft_phase'] = np.angle(stft)
            
            # Additional STFT resolutions for album analysis
            try:
                stft_short = librosa.stft(y, hop_length=self.hop_length//2, n_fft=self.n_fft//2)  # Higher time resolution
                features['stft_short_magnitude'] = np.abs(stft_short)
                
                stft_long = librosa.stft(y, hop_length=self.hop_length*2, n_fft=self.n_fft*2)  # Higher freq resolution  
                features['stft_long_magnitude'] = np.abs(stft_long)
            except:
                features['stft_short_magnitude'] = None
                features['stft_long_magnitude'] = None
            
            # Reconstruct audio from STFT
            # NOTE: Audio reconstruction removed for file size optimization (~42MB per track)
            # To re-enable for other applications, uncomment this line:
            # features['audio_reconstructed'] = librosa.istft(stft, hop_length=self.hop_length)
            
            # Constant-Q Transform
            try:
                cqt = librosa.cqt(y, sr=sr, hop_length=self.hop_length)
                features['cqt_real'] = np.real(cqt)
                features['cqt_imag'] = np.imag(cqt)
                features['cqt_magnitude'] = np.abs(cqt)
                features['cqt_phase'] = np.angle(cqt)
            except Exception as e:
                print(f"    CQT failed: {e}")
                features['cqt_real'] = None
            
            # Variable-Q Transform
            # NOTE: VQT removed for file size optimization (~500MB) - redundant with CQT/STFT
            # To re-enable for other applications, uncomment these lines:
            # try:
            #     vqt = librosa.vqt(y, sr=sr, hop_length=self.hop_length)
            #     features['vqt_magnitude'] = np.abs(vqt)
            # except:
            #     features['vqt_magnitude'] = None
            
            # Phase recovery
            # NOTE: Griffin-Lim reconstruction removed for file size optimization (~42MB per track)
            # To re-enable for other applications, uncomment these lines:
            # try:
            #     features['griffinlim_reconstruction'] = librosa.griffinlim(features['stft_magnitude'])
            # except:
            #     features['griffinlim_reconstruction'] = None
            
            # Magnitude scaling
            features['stft_db'] = librosa.amplitude_to_db(features['stft_magnitude'])
            features['power_db'] = librosa.power_to_db(features['stft_magnitude']**2)
            
            # PCEN normalization
            try:
                features['pcen'] = librosa.pcen(features['stft_magnitude']**2)
            except:
                features['pcen'] = None
            
            # Frequency arrays
            features['fft_frequencies'] = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            features['mel_frequencies'] = librosa.mel_frequencies(n_mels=self.n_mels, fmax=sr/2)
            
            # Pitch tracking
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                features['f0_pyin'] = f0
                features['voiced_flag'] = voiced_flag
                features['voiced_probabilities'] = voiced_probs
            except:
                features['f0_pyin'] = None
            
            try:
                pitches, magnitudes = librosa.piptrack(S=features['stft_magnitude'], sr=sr)
                features['piptrack_pitches'] = pitches
                features['piptrack_magnitudes'] = magnitudes
            except:
                features['piptrack_pitches'] = None
            
            # === FEATURE EXTRACTION ===
            print("  Extracting spectral features...")
            
            # Basic spectral features
            features['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=sr)
            try:
                features['chroma_cqt'] = librosa.feature.chroma_cqt(y=y, sr=sr)
            except:
                features['chroma_cqt'] = None
            
            # Enhanced chroma variants
            try:
                features['chroma_cens'] = librosa.feature.chroma_cens(y=y, sr=sr)
                features['chroma_deep'] = librosa.feature.chroma_stft(y=y, sr=sr, norm=np.inf, threshold=0.0)
            except:
                features['chroma_cens'] = None
                features['chroma_deep'] = None
                
            features['melspectrogram'] = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features['rms'] = librosa.feature.rms(y=y)
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_contrast'] = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_flatness'] = librosa.feature.spectral_flatness(y=y)
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['poly_features'] = librosa.feature.poly_features(S=features['stft_magnitude'], sr=sr)
            features['tonnetz'] = librosa.feature.tonnetz(y=y, sr=sr)
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)
            
            # Rhythm features
            print("  Extracting rhythm features...")
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = tempo
                features['beats'] = beats
                
                # Beat interval analysis for tempo stability
                if len(beats) > 1:
                    beat_times = librosa.frames_to_time(beats, sr=sr)
                    beat_intervals = np.diff(beat_times)
                    features['tempo_variance'] = np.var(beat_intervals)
                    features['tempo_stability'] = 1.0 / (1.0 + features['tempo_variance'])  # Higher = more stable
                else:
                    features['tempo_variance'] = None
                    features['tempo_stability'] = None
                    
            except:
                features['tempo'] = None
                features['beats'] = None
                features['tempo_variance'] = None
                features['tempo_stability'] = None
            
            features['tempogram'] = librosa.feature.tempogram(y=y, sr=sr)
            features['fourier_tempogram'] = librosa.feature.fourier_tempogram(y=y, sr=sr)
            
            # Feature manipulation
            features['mfcc_delta'] = librosa.feature.delta(features['mfcc'])
            features['mfcc_delta2'] = librosa.feature.delta(features['mfcc'], order=2)
            
            # === BEAT-SYNCHRONOUS FEATURES (CRUCIAL FOR ALBUM ARRANGEMENT) ===
            print("  Extracting beat-synchronous features...")
            if features['beats'] is not None and len(features['beats']) > 0:
                try:
                    # Synchronize key features to beat grid
                    features['mfcc_beat_sync'] = librosa.util.sync(features['mfcc'], features['beats'])
                    features['chroma_beat_sync'] = librosa.util.sync(features['chroma_stft'], features['beats'])
                    features['spectral_centroid_beat_sync'] = librosa.util.sync(features['spectral_centroid'], features['beats'])
                    features['rms_beat_sync'] = librosa.util.sync(features['rms'], features['beats'])
                    
                    # Beat-wise tempo analysis
                    beat_times = librosa.frames_to_time(features['beats'], sr=sr)
                    features['beat_times'] = beat_times
                    if len(beat_times) > 2:
                        features['local_tempo'] = 60.0 / np.diff(beat_times)  # BPM for each beat interval
                except:
                    features['mfcc_beat_sync'] = None
                    
            # === ONSET DETECTION ===
            print("  Extracting onset features...")
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength'] = onset_strength
            features['onsets'] = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr)
            
            # Spectral flux (alternative onset measure)
            if features['stft_magnitude'].shape[1] > 1:
                spectral_flux = np.sum(np.maximum(0, np.diff(features['stft_magnitude'], axis=1))**2, axis=0)
                features['spectral_flux'] = spectral_flux
            
            # === DECOMPOSITION ===
            print("  Extracting decomposition features...")
            try:
                # Harmonic-percussive separation
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                # NOTE: Raw separated audio removed for file size optimization (~84MB per track)
                # To re-enable for other applications, uncomment these lines:
                # features['audio_harmonic'] = y_harmonic
                # features['audio_percussive'] = y_percussive
                
                # Spectral decomposition
                S_harmonic, S_percussive = librosa.decompose.hpss(features['stft_magnitude'])
                features['stft_harmonic'] = S_harmonic
                features['stft_percussive'] = S_percussive
                
                # Harmonic change detection (for key/chord changes)
                features['harmonic_change'] = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
                
            except:
                # Raw audio components not stored - commented out for file size optimization
                features['harmonic_change'] = None
            
            # === SEGMENTATION ===
            print("  Extracting segmentation features...")
            try:
                # Use beat-synchronized MFCC for better segmentation
                mfcc_for_segmentation = features['mfcc_beat_sync'] if features.get('mfcc_beat_sync') is not None else features['mfcc']
                
                if mfcc_for_segmentation.shape[1] > 10:  # Only if we have enough frames
                    R = librosa.segment.recurrence_matrix(mfcc_for_segmentation, mode='connectivity')
                    features['recurrence_matrix'] = R
                    
                    # Structural segmentation
                    try:
                        boundaries = librosa.segment.agglomerative(R, k=None)
                        features['structure_boundaries'] = boundaries
                        
                        # Convert boundaries to time
                        if features['beats'] is not None:
                            boundary_times = librosa.frames_to_time(features['beats'][boundaries], sr=sr)
                        else:
                            boundary_times = librosa.frames_to_time(boundaries * self.hop_length, sr=sr)
                        features['structure_boundary_times'] = boundary_times
                        
                    except:
                        features['structure_boundaries'] = None
                        features['structure_boundary_times'] = None
                else:
                    features['recurrence_matrix'] = None
                    features['structure_boundaries'] = None
                    
            except:
                features['recurrence_matrix'] = None
                features['structure_boundaries'] = None
            
            # === ALBUM-SPECIFIC ANALYSIS ===
            print("  Extracting album-specific features...")
            
            # Energy trajectory analysis (crucial for album pacing)
            try:
                # Smooth energy evolution over time
                window_size = int(sr / self.hop_length * 10)  # 10-second windows
                if len(features['rms'][0]) > window_size:
                    features['energy_trajectory'] = np.convolve(features['rms'][0], 
                                                               np.ones(window_size)/window_size, 
                                                               mode='valid')
                else:
                    features['energy_trajectory'] = features['rms'][0]
                    
                # Energy statistics for album balancing
                features['energy_mean'] = np.mean(features['rms'])
                features['energy_std'] = np.std(features['rms'])
                features['energy_max'] = np.max(features['rms'])
                features['energy_min'] = np.min(features['rms'])
                
                # Dynamic range (crucial for mastering consistency)
                features['dynamic_range'] = librosa.amplitude_to_db(features['energy_max']) - librosa.amplitude_to_db(features['energy_min'])
                
                # Loudness approximation (psychoacoustic)
                features['loudness_rms'] = np.sqrt(np.mean(features['rms']**2))
                
            except:
                features['energy_trajectory'] = None
                features['dynamic_range'] = None
            
            # Silence detection and analysis
            try:
                # Find silent regions (useful for natural transitions)
                non_silent_intervals = librosa.effects.split(y, top_db=20)
                features['silence_intervals'] = non_silent_intervals
                
                # Calculate silence statistics
                if len(non_silent_intervals) > 0:
                    total_samples = len(y)
                    non_silent_samples = np.sum([end - start for start, end in non_silent_intervals])
                    features['silence_ratio'] = 1.0 - (non_silent_samples / total_samples)
                else:
                    features['silence_ratio'] = 1.0  # Completely silent
                    
                # Fade-in/fade-out detection
                fade_samples = int(0.1 * sr)  # 100ms
                if len(y) > 2 * fade_samples:
                    features['fade_in_energy'] = np.mean(features['rms'][0][:fade_samples//self.hop_length])
                    features['fade_out_energy'] = np.mean(features['rms'][0][-fade_samples//self.hop_length:])
                else:
                    features['fade_in_energy'] = None
                    features['fade_out_energy'] = None
                    
            except:
                features['silence_intervals'] = None
                features['silence_ratio'] = None
            
            # Perceptual features
            try:
                # Spectral roughness (perceived harshness)
                if features['spectral_bandwidth'] is not None and features['spectral_centroid'] is not None:
                    features['spectral_roughness'] = np.mean(features['spectral_bandwidth'] / (features['spectral_centroid'] + 1e-8))
                else:
                    features['spectral_roughness'] = None
                    
                # Tonal vs. noise content
                features['tonality_ratio'] = np.mean(features['spectral_flatness'])  # Lower = more tonal
                
            except:
                features['spectral_roughness'] = None
                features['tonality_ratio'] = None
            
            # Key and mode estimation (for harmonic compatibility)
            try:
                if features['chroma_cqt'] is not None:
                    # Simple key estimation using chroma profile correlation
                    chroma_mean = np.mean(features['chroma_cqt'], axis=1)
                    features['chroma_profile'] = chroma_mean
                    
                    # Major/minor mode estimation
                    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major template
                    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # C minor template
                    
                    major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
                    minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
                    
                    features['major_correlation'] = major_corr
                    features['minor_correlation'] = minor_corr
                    features['estimated_mode'] = 'major' if major_corr > minor_corr else 'minor'
                    
            except:
                features['chroma_profile'] = None
                features['estimated_mode'] = None
            
            # === SEQUENTIAL MODELING ===
            print("  Extracting sequential features...")
            try:
                n_states = 12  # Use 12 states for chroma-like analysis
                features['transition_uniform'] = librosa.sequence.transition_uniform(n_states)
                features['transition_loop'] = librosa.sequence.transition_loop(n_states, prob=0.9)
                features['transition_cycle'] = librosa.sequence.transition_cycle(n_states)
            except:
                features['transition_uniform'] = None
            
            # === UTILITY ARRAYS ===
            print("  Extracting utility features...")
            
            # Time and frame conversions
            frames = librosa.frames_to_time(np.arange(features['stft_magnitude'].shape[1]), sr=sr, hop_length=self.hop_length)
            features['frame_times'] = frames
            
            # Peak picking on onset strength
            if onset_strength is not None:
                features['onset_peaks'] = librosa.util.peak_pick(onset_strength, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
            
            # === FILTER BANKS ===
            print("  Extracting filter banks...")
            features['mel_filter_bank'] = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.n_mels)
            features['chroma_filter_bank'] = librosa.filters.chroma(sr=sr, n_fft=self.n_fft, n_chroma=self.n_chroma)
            
            # Window function
            features['hann_window'] = librosa.filters.get_window('hann', self.n_fft)
            
            # === SUMMARY STATISTICS FOR ALBUM ARRANGEMENT ===
            print("  Computing summary statistics...")
            
            # Create summary stats for key features (useful for quick comparisons)
            summary_features = ['tempo', 'energy_mean', 'spectral_centroid', 'spectral_bandwidth', 
                               'spectral_rolloff', 'zero_crossing_rate', 'mfcc']
            
            for feat_name in summary_features:
                if feat_name in features and features[feat_name] is not None:
                    feat_data = features[feat_name]
                    if feat_data.ndim > 1:  # Multi-dimensional features like MFCC
                        for i in range(feat_data.shape[0]):
                            features[f'{feat_name}_mean_{i}'] = np.mean(feat_data[i])
                            features[f'{feat_name}_std_{i}'] = np.std(feat_data[i])
                    else:  # 1D features
                        if isinstance(feat_data, (int, float)):
                            features[f'{feat_name}_value'] = feat_data
                        else:
                            features[f'{feat_name}_mean'] = np.mean(feat_data)
                            features[f'{feat_name}_std'] = np.std(feat_data)
            
            print(f"  Successfully extracted {len([k for k, v in features.items() if v is not None])} features")
            
            # Add rebuild information for ChatGPT reference
            features['rebuild_info'] = {
                'removed_for_size': [
                    'audio_raw', 'audio_reconstructed', 'griffinlim_reconstruction',
                    'audio_harmonic', 'audio_percussive', 'click_track',
                    'stft_real', 'stft_imag', 'cqt_real', 'cqt_imag', 'vqt_magnitude',
                    'mel_filter_bank', 'chroma_filter_bank', 'hann_window',
                    'transition_uniform', 'transition_loop', 'transition_cycle', 'recurrence_matrix'
                ],
                'rebuild_commands': {
                    'stft_complex': 'stft_real + 1j*stft_imag can be rebuilt from magnitude*exp(1j*phase)',
                    'cqt_complex': 'cqt_real + 1j*cqt_imag can be rebuilt from magnitude*exp(1j*phase)',
                    'mel_filter_bank': f'librosa.filters.mel(sr={sr}, n_fft={self.n_fft}, n_mels={self.n_mels})',
                    'chroma_filter_bank': f'librosa.filters.chroma(sr={sr}, n_fft={self.n_fft}, n_chroma={self.n_chroma})',
                    'hann_window': f'librosa.filters.get_window("hann", {self.n_fft})',
                    'transition_uniform': 'librosa.sequence.transition_uniform(12)',
                    'transition_loop': 'librosa.sequence.transition_loop(12, prob=0.9)',
                    'transition_cycle': 'librosa.sequence.transition_cycle(12)',
                    'recurrence_matrix': 'librosa.segment.recurrence_matrix(mfcc_beat_sync, mode="connectivity")',
                    'click_track': f'librosa.clicks(times=click_times, sr={sr}, length=audio_length)'
                },
                'parameters': {
                    'sr': sr,
                    'hop_length': self.hop_length,
                    'n_fft': self.n_fft,
                    'n_mels': self.n_mels,
                    'n_chroma': self.n_chroma
                },
                'note': 'Matrices and raw audio removed to reduce file size by ~5-8GB per track'
            }
            
            return features
            
        except Exception as e:
            print(f"  ERROR processing {audio_path}: {e}")
            traceback.print_exc()
            return None
    
    def process_folder(self, input_folder, output_folder=None):
        """Process all audio files in a folder"""
        input_path = Path(input_folder)
        if output_folder is None:
            output_path = input_path / "features"
        else:
            output_path = Path(output_folder)
        
        output_path.mkdir(exist_ok=True)
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.aif', '.m4a', '.ogg', '.wma'}
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(audio_files)} audio files in {input_folder}")
        print(f"Output folder: {output_path}")
        
        success_count = 0
        for audio_file in audio_files:
            try:
                # Extract features
                features = self.extract_all_features(str(audio_file))
                
                if features is not None:
                    # Save as compressed numpy file
                    output_file = output_path / f"{audio_file.stem}_features.npz"
                    
                    # Filter out None values and convert to serializable formats
                    # NOTE: Float precision optimization for file size vs. accuracy balance
                    # - float64 (default): 8 bytes, ~15 digits precision - overkill for audio analysis
                    # - float32 (chosen): 4 bytes, ~7 digits precision - perfect for musical features
                    # - float16: 2 bytes, ~3 digits precision - too lossy for detailed analysis
                    # To use different precision, change .astype(np.float32) below to desired type
                    clean_features = {}
                    for key, value in features.items():
                        if value is not None:
                            if isinstance(value, np.ndarray):
                                # Convert to float32 for file size optimization
                                if value.dtype in [np.float64, np.complex128]:
                                    if np.iscomplexobj(value):
                                        clean_features[key] = value.astype(np.complex64)
                                    else:
                                        clean_features[key] = value.astype(np.float32)
                                else:
                                    clean_features[key] = value
                            elif isinstance(value, (int, float, complex)):
                                clean_features[key] = np.array(value, dtype=np.float32)
                            elif isinstance(value, str):
                                # Store strings as arrays of unicode code points for npz compatibility
                                clean_features[key] = np.array([ord(c) for c in value])
                            elif isinstance(value, dict):
                                # Handle rebuild_info dictionary
                                clean_features[key] = value
                            elif isinstance(value, list):
                                try:
                                    arr = np.array(value)
                                    if arr.dtype in [np.float64, np.complex128]:
                                        if np.iscomplexobj(arr):
                                            clean_features[key] = arr.astype(np.complex64)
                                        else:
                                            clean_features[key] = arr.astype(np.float32)
                                    else:
                                        clean_features[key] = arr
                                except:
                                    pass  # Skip if can't convert to array
                    
                    np.savez_compressed(output_file, **clean_features)
                    print(f"  Saved: {output_file}")
                    success_count += 1
                
            except Exception as e:
                print(f"  Failed to process {audio_file}: {e}")
                continue
        
        print(f"\nCompleted! Successfully processed {success_count}/{len(audio_files)} files")
        print(f"Features saved in: {output_path}")
        
        # Print a sample of what was extracted
        if success_count > 0:
            sample_file = list(output_path.glob("*_features.npz"))[0]
            sample_data = np.load(sample_file)
            print(f"\nSample feature file: {sample_file.name}")
            print(f"Contains {len(sample_data.keys())} features:")
            
            # Group features by category for better display
            categories = {
                'Basic': ['duration', 'tempo', 'sample_rate'],
                'Energy': [k for k in sample_data.keys() if 'energy' in k or 'rms' in k or 'loudness' in k],
                'Spectral': [k for k in sample_data.keys() if 'spectral' in k or 'mfcc' in k or 'chroma' in k],
                'Rhythm': [k for k in sample_data.keys() if 'beat' in k or 'tempo' in k or 'onset' in k],
                'Structure': [k for k in sample_data.keys() if 'boundary' in k or 'segment' in k or 'recurrence' in k],
                'Album': [k for k in sample_data.keys() if any(x in k for x in ['trajectory', 'fade', 'silence', 'dynamic_range', 'stability'])]
            }
            
            for category, keys in categories.items():
                matching_keys = [k for k in keys if k in sample_data.keys()]
                if matching_keys:
                    print(f"\n  {category} features ({len(matching_keys)}):")
                    for key in matching_keys[:3]:  # Show first 3 in each category
                        shape = sample_data[key].shape if hasattr(sample_data[key], 'shape') else 'scalar'
                        print(f"    {key}: {shape}")
                    if len(matching_keys) > 3:
                        print(f"    ... and {len(matching_keys) - 3} more")

def main():
    parser = argparse.ArgumentParser(description='Extract enhanced librosa features for album analysis')
    parser.add_argument('input_folder', help='Folder containing audio files')
    parser.add_argument('output_folder', nargs='?', help='Output folder for feature files (default: input_folder/features)')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate (default: 22050)')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length (default: 512)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size (default: 2048)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    # Create extractor
    extractor = EnhancedLibrosaFeatureExtractor(sr=args.sr, hop_length=args.hop_length, n_fft=args.n_fft)
    
    # Process folder
    extractor.process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
