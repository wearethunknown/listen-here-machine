#!/usr/bin/env python3
"""
Comprehensive Librosa Feature Extractor

This script scans a folder for audio files and extracts ALL possible librosa features,
saving them as compressed .npz files with the same base filename as the audio tracks.

Usage: python librosa_feature_extractor.py input_folder [output_folder]

Requirements:
    pip install librosa numpy

Author: Generated for comprehensive audio feature extraction
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class LibrosaFeatureExtractor:
    def __init__(self, sr=22050, hop_length=512, n_fft=2048):
        """Initialize with default parameters"""
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = 128
        self.n_mfcc = 13
        self.n_chroma = 12
        
    def extract_all_features(self, audio_path):
        """Extract all possible librosa features from an audio file"""
        print(f"Processing: {audio_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            features = {}
            
            # Store basic audio info
            features['audio_raw'] = y
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
            
            # Spectral representations
            stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
            features['stft_real'] = np.real(stft)
            features['stft_imag'] = np.imag(stft)
            features['stft_magnitude'] = np.abs(stft)
            features['stft_phase'] = np.angle(stft)
            
            # Reconstruct audio from STFT
            features['audio_reconstructed'] = librosa.istft(stft, hop_length=self.hop_length)
            
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
            
            # Phase recovery
            try:
                features['griffinlim_reconstruction'] = librosa.griffinlim(features['stft_magnitude'])
            except:
                features['griffinlim_reconstruction'] = None
            
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
            
            # Spectral features
            features['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=sr)
            try:
                features['chroma_cqt'] = librosa.feature.chroma_cqt(y=y, sr=sr)
            except:
                features['chroma_cqt'] = None
                
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
            except:
                features['tempo'] = None
                features['beats'] = None
            
            features['tempogram'] = librosa.feature.tempogram(y=y, sr=sr)
            features['fourier_tempogram'] = librosa.feature.fourier_tempogram(y=y, sr=sr)
            
            # Feature manipulation
            features['mfcc_delta'] = librosa.feature.delta(features['mfcc'])
            features['mfcc_delta2'] = librosa.feature.delta(features['mfcc'], order=2)
            
            # === ONSET DETECTION ===
            print("  Extracting onset features...")
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength'] = onset_strength
            features['onsets'] = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr)
            
            # === DECOMPOSITION ===
            print("  Extracting decomposition features...")
            try:
                # Harmonic-percussive separation
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                features['audio_harmonic'] = y_harmonic
                features['audio_percussive'] = y_percussive
                
                # Spectral decomposition
                S_harmonic, S_percussive = librosa.decompose.hpss(features['stft_magnitude'])
                features['stft_harmonic'] = S_harmonic
                features['stft_percussive'] = S_percussive
            except:
                features['audio_harmonic'] = None
            
            # === SEGMENTATION ===
            print("  Extracting segmentation features...")
            try:
                # Recurrence matrix (use MFCC for efficiency)
                mfcc_sync = librosa.util.sync(features['mfcc'], features['beats']) if features['beats'] is not None else features['mfcc']
                if mfcc_sync.shape[1] > 10:  # Only if we have enough frames
                    features['recurrence_matrix'] = librosa.segment.recurrence_matrix(mfcc_sync, mode='connectivity')
                else:
                    features['recurrence_matrix'] = None
            except:
                features['recurrence_matrix'] = None
            
            # === SEQUENTIAL MODELING ===
            print("  Extracting sequential features...")
            # Create simple transition matrices for reference
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
            
            print(f"  Successfully extracted {len([k for k, v in features.items() if v is not None])} features")
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
                    clean_features = {}
                    for key, value in features.items():
                        if value is not None:
                            if isinstance(value, np.ndarray):
                                clean_features[key] = value
                            elif isinstance(value, (int, float, complex)):
                                clean_features[key] = np.array(value)
                            elif isinstance(value, list):
                                try:
                                    clean_features[key] = np.array(value)
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
            for key in sorted(sample_data.keys())[:10]:  # Show first 10
                shape = sample_data[key].shape if hasattr(sample_data[key], 'shape') else 'scalar'
                print(f"  {key}: {shape}")
            if len(sample_data.keys()) > 10:
                print(f"  ... and {len(sample_data.keys()) - 10} more")

def main():
    parser = argparse.ArgumentParser(description='Extract all librosa features from audio files')
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
    extractor = LibrosaFeatureExtractor(sr=args.sr, hop_length=args.hop_length, n_fft=args.n_fft)
    
    # Process folder
    extractor.process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
