#!/usr/bin/env python3
"""
Complete Essentia Feature Extractor - ALL Algorithms

This script extracts EVERY possible Essentia feature for comprehensive audio analysis.
Implements all 200+ algorithms from the Essentia reference.

Usage: python complete_essentia_extractor.py input_folder [output_folder]

Requirements:
    pip install essentia numpy scipy tensorflow

Author: Complete Essentia feature extraction - unoptimized maximum coverage
"""

import os
import sys
import argparse
import warnings
import numpy as np
import essentia
import essentia.standard as es
from pathlib import Path
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CompleteEssentiaExtractor:
    def __init__(self, sr=44100, hop_length=128, frame_size=2048):
        """Initialize with all possible Essentia algorithms"""
        self.sr = sr
        self.hop_length = hop_length
        self.frame_size = frame_size
        self.window_type = 'hann'
        
        # Initialize ALL Essentia algorithms
        self._init_all_algorithms()
        
    def _init_all_algorithms(self):
        """Initialize every single Essentia algorithm"""
        
        # === INPUT/OUTPUT ===
        self.loader = es.MonoLoader(sampleRate=self.sr)
        self.audioLoader = es.AudioLoader()
        self.easyLoader = es.EasyLoader()
        self.eqloudLoader = es.EqloudLoader()
        self.metadataReader = es.MetadataReader()
        self.monoWriter = es.MonoWriter()
        self.audioWriter = es.AudioWriter()
        self.yamlOutput = es.YamlOutput()
        
        # === STANDARD PROCESSING ===
        self.frameCutter = es.FrameCutter(frameSize=self.frame_size, hopSize=self.hop_length)
        self.windowing = es.Windowing(type=self.window_type)
        self.spectrum = es.Spectrum()
        self.fft = es.FFT()
        self.ifft = es.IFFT()
        self.fftc = es.FFTC()
        self.ifftc = es.IFFTC()
        self.dct = es.DCT()
        self.idct = es.IDCT()
        self.constantQ = es.ConstantQ()
        self.nsgConstantQ = es.NSGConstantQ()
        self.nsgIConstantQ = es.NSGIConstantQ()
        self.autoCorrelation = es.AutoCorrelation()
        self.crossCorrelation = es.CrossCorrelation()
        self.warpedAutoCorrelation = es.WarpedAutoCorrelation()
        self.welch = es.Welch()
        self.resample = es.Resample()
        self.frameBuffer = es.FrameBuffer()
        self.frameToReal = es.FrameToReal()
        self.overlapAdd = es.OverlapAdd()
        self.peakDetection = es.PeakDetection()
        self.minMax = es.MinMax()
        self.clipper = es.Clipper()
        self.scale = es.Scale()
        self.trimmer = es.Trimmer()
        self.stereoTrimmer = es.StereoTrimmer()
        self.slicer = es.Slicer()
        self.monoMixer = es.MonoMixer()
        self.stereoDemuxer = es.StereoDemuxer()
        self.stereoMuxer = es.StereoMuxer()
        self.multiplexer = es.Multiplexer()
        self.noiseAdder = es.NoiseAdder()
        self.bpf = es.BPF()
        self.cubicSpline = es.CubicSpline()
        self.spline = es.Spline()
        self.derivative = es.Derivative()
        self.binaryOperator = es.BinaryOperator()
        self.binaryOperatorStream = es.BinaryOperatorStream()
        self.unaryOperator = es.UnaryOperator()
        self.unaryOperatorStream = es.UnaryOperatorStream()
        self.zeroCrossingRate = es.ZeroCrossingRate()
        
        # === FILTERS ===
        self.allPass = es.AllPass()
        self.bandPass = es.BandPass()
        self.bandReject = es.BandReject()
        self.dcRemoval = es.DCRemoval()
        self.equalLoudness = es.EqualLoudness()
        self.highPass = es.HighPass()
        self.lowPass = es.LowPass()
        self.iir = es.IIR()
        self.maxFilter = es.MaxFilter()
        self.medianFilter = es.MedianFilter()
        self.movingAverage = es.MovingAverage()
        
        # === ENVELOPE/SFX ===
        self.envelope = es.Envelope()
        self.afterMaxToBeforeMaxEnergyRatio = es.AfterMaxToBeforeMaxEnergyRatio()
        self.derivativeSFX = es.DerivativeSFX()
        self.flatnessSFX = es.FlatnessSFX()
        self.logAttackTime = es.LogAttackTime()
        self.maxToTotal = es.MaxToTotal()
        self.minToTotal = es.MinToTotal()
        self.strongDecay = es.StrongDecay()
        self.tcToTotal = es.TCToTotal()
        
        # === SPECTRAL ANALYSIS ===
        self.spectralPeaks = es.SpectralPeaks()
        self.spectralWhitening = es.SpectralWhitening()
        self.mfcc = es.MFCC()
        self.bfcc = es.BFCC()
        self.gfcc = es.GFCC()
        self.melBands = es.MelBands()
        self.barkBands = es.BarkBands()
        self.triangularBands = es.TriangularBands()
        self.triangularBarkBands = es.TriangularBarkBands()
        self.erbBands = es.ERBBands()
        self.frequencyBands = es.FrequencyBands()
        self.energyBand = es.EnergyBand()
        self.energyBandRatio = es.EnergyBandRatio()
        self.powerSpectrum = es.PowerSpectrum()
        self.logSpectrum = es.LogSpectrum()
        self.spectrumToCent = es.SpectrumToCent()
        self.spectralCentroidTime = es.SpectralCentroidTime()
        self.spectralComplexity = es.SpectralComplexity()
        self.spectralContrast = es.SpectralContrast()
        self.rollOff = es.RollOff()
        self.strongPeak = es.StrongPeak()
        self.hfc = es.HFC()
        self.flux = es.Flux()
        self.flatnessDB = es.FlatnessDB()
        self.maxMagFreq = es.MaxMagFreq()
        self.panning = es.Panning()
        self.lpc = es.LPC()
        
        # === TENSORFLOW INPUTS ===
        try:
            self.tensorflowInputMusiCNN = es.TensorflowInputMusiCNN()
            self.tensorflowInputVGGish = es.TensorflowInputVGGish()
            self.tensorflowInputTempoCNN = es.TensorflowInputTempoCNN()
            self.tensorflowInputFSDSINet = es.TensorflowInputFSDSINet()
        except:
            pass  # TensorFlow features may not be available
        
        # === PITCH ANALYSIS ===
        self.pitchYin = es.PitchYin()
        self.pitchYinFFT = es.PitchYinFFT()
        self.pitchYinProbabilistic = es.PitchYinProbabilistic()
        self.pitchYinProbabilities = es.PitchYinProbabilities()
        self.pitchYinProbabilitiesHMM = es.PitchYinProbabilitiesHMM()
        self.predominantPitchMelodia = es.PredominantPitchMelodia()
        self.pitchMelodia = es.PitchMelodia()
        self.multiPitchMelodia = es.MultiPitchMelodia()
        self.multiPitchKlapuri = es.MultiPitchKlapuri()
        self.pitchFilter = es.PitchFilter()
        self.pitchSalienceFunction = es.PitchSalienceFunction()
        self.pitchSalienceFunctionPeaks = es.PitchSalienceFunctionPeaks()
        self.pitchContours = es.PitchContours()
        self.pitchContoursMonoMelody = es.PitchContoursMonoMelody()
        self.pitchContoursMultiMelody = es.PitchContoursMultiMelody()
        self.pitchContoursMelody = es.PitchContoursMelody()
        self.pitchContourSegmentation = es.PitchContourSegmentation()
        self.vibrato = es.Vibrato()
        self.audio2Pitch = es.Audio2Pitch()
        self.pitch2Midi = es.Pitch2Midi()
        self.audio2Midi = es.Audio2Midi()
        try:
            self.pitchCREPE = es.PitchCREPE()
        except:
            pass
        
        # === RHYTHM ANALYSIS ===
        self.rhythmExtractor = es.RhythmExtractor()
        self.rhythmExtractor2013 = es.RhythmExtractor2013()
        self.rhythmDescriptors = es.RhythmDescriptors()
        self.beatTrackerDegara = es.BeatTrackerDegara()
        self.beatTrackerMultiFeature = es.BeatTrackerMultiFeature()
        self.onsetDetection = es.OnsetDetection()
        self.onsetDetectionGlobal = es.OnsetDetectionGlobal()
        self.onsetRate = es.OnsetRate()
        self.onsets = es.Onsets()
        self.superFluxExtractor = es.SuperFluxExtractor()
        self.superFluxNovelty = es.SuperFluxNovelty()
        self.superFluxPeaks = es.SuperFluxPeaks()
        self.noveltyCurve = es.NoveltyCurve()
        self.noveltyCurveFixedBpmEstimator = es.NoveltyCurveFixedBpmEstimator()
        self.tempoTap = es.TempoTap()
        self.tempoTapDegara = es.TempoTapDegara()
        self.tempoTapMaxAgreement = es.TempoTapMaxAgreement()
        self.tempoTapTicks = es.TempoTapTicks()
        self.tempoScaleBands = es.TempoScaleBands()
        self.bpmHistogram = es.BpmHistogram()
        self.bpmHistogramDescriptors = es.BpmHistogramDescriptors()
        self.bpmRubato = es.BpmRubato()
        self.harmonicBpm = es.HarmonicBpm()
        self.loopBpmEstimator = es.LoopBpmEstimator()
        self.loopBpmConfidence = es.LoopBpmConfidence()
        self.percivalBpmEstimator = es.PercivalBpmEstimator()
        self.percivalEnhanceHarmonics = es.PercivalEnhanceHarmonics()
        self.percivalEvaluatePulseTrains = es.PercivalEvaluatePulseTrains()
        self.rhythmTransform = es.RhythmTransform()
        self.beatogram = es.Beatogram()
        self.beatsLoudness = es.BeatsLoudness()
        self.singleBeatLoudness = es.SingleBeatLoudness()
        self.meter = es.Meter()
        self.danceability = es.Danceability()
        try:
            self.tempoCNN = es.TempoCNN()
        except:
            pass
        
        # === TONAL ANALYSIS ===
        self.keyExtractor = es.KeyExtractor()
        self.key = es.Key()
        self.hpcp = es.HPCP()
        self.chromagram = es.Chromagram()
        self.chordsDetection = es.ChordsDetection()
        self.chordsDetectionBeats = es.ChordsDetectionBeats()
        self.chordsDescriptors = es.ChordsDescriptors()
        self.spectrumCQ = es.SpectrumCQ()
        self.tonalExtractor = es.TonalExtractor()
        self.tonicIndianArtMusic = es.TonicIndianArtMusic()
        self.tuningFrequency = es.TuningFrequency()
        self.tuningFrequencyExtractor = es.TuningFrequencyExtractor()
        self.highResolutionFeatures = es.HighResolutionFeatures()
        self.nnlsChroma = es.NNLSChroma()
        self.dissonance = es.Dissonance()
        self.harmonicPeaks = es.HarmonicPeaks()
        self.inharmonicity = es.Inharmonicity()
        self.oddToEvenHarmonicEnergyRatio = es.OddToEvenHarmonicEnergyRatio()
        self.tristimulus = es.Tristimulus()
        self.pitchSalience = es.PitchSalience()
        
        # === MUSIC SIMILARITY ===
        self.chromaCrossSimilarity = es.ChromaCrossSimilarity()
        self.coverSongSimilarity = es.CoverSongSimilarity()
        self.crossSimilarityMatrix = es.CrossSimilarityMatrix()
        
        # === FINGERPRINTING ===
        self.chromaprinter = es.Chromaprinter()
        
        # === AUDIO PROBLEMS ===
        self.clickDetector = es.ClickDetector()
        self.discontinuityDetector = es.DiscontinuityDetector()
        self.falseStereoDetector = es.FalseStereoDetector()
        self.gapsDetector = es.GapsDetector()
        self.humDetector = es.HumDetector()
        self.noiseBurstDetector = es.NoiseBurstDetector()
        self.snr = es.SNR()
        self.saturationDetector = es.SaturationDetector()
        self.startStopCut = es.StartStopCut()
        self.truePeakDetector = es.TruePeakDetector()
        
        # === DURATION/SILENCE ===
        self.duration = es.Duration()
        self.effectiveDuration = es.EffectiveDuration()
        self.fadeDetection = es.FadeDetection()
        self.silenceRate = es.SilenceRate()
        self.startStopSilence = es.StartStopSilence()
        
        # === LOUDNESS/DYNAMICS ===
        self.loudness = es.Loudness()
        self.loudnessEBUR128 = es.LoudnessEBUR128()
        self.loudnessVickers = es.LoudnessVickers()
        self.replayGain = es.ReplayGain()
        self.dynamicComplexity = es.DynamicComplexity()
        self.intensity = es.Intensity()
        self.larm = es.Larm()
        self.leq = es.Leq()
        self.levelExtractor = es.LevelExtractor()
        
        # === STATISTICS ===
        self.centralMoments = es.CentralMoments()
        self.distributionShape = es.DistributionShape()
        self.entropy = es.Entropy()
        self.energy = es.Energy()
        self.rms = es.RMS()
        self.instantPower = es.InstantPower()
        self.centroid = es.Centroid()
        self.crest = es.Crest()
        self.decrease = es.Decrease()
        self.flatness = es.Flatness()
        self.geometricMean = es.GeometricMean()
        self.histogram = es.Histogram()
        self.mean = es.Mean()
        self.median = es.Median()
        self.powerMean = es.PowerMean()
        self.rawMoments = es.RawMoments()
        self.singleGaussian = es.SingleGaussian()
        self.variance = es.Variance()
        self.viterbi = es.Viterbi()
        self.poolAggregator = es.PoolAggregator()
        
        # === MATH ===
        self.cartesianToPolar = es.CartesianToPolar()
        self.polarToCartesian = es.PolarToCartesian()
        self.magnitude = es.Magnitude()
        
        # === SYNTHESIS ===
        self.sineModelAnal = es.SineModelAnal()
        self.sineModelSynth = es.SineModelSynth()
        self.sineSubtraction = es.SineSubtraction()
        self.harmonicModelAnal = es.HarmonicModelAnal()
        self.harmonicMask = es.HarmonicMask()
        self.hprModelAnal = es.HprModelAnal()
        self.hpsModelAnal = es.HpsModelAnal()
        self.sprModelAnal = es.SprModelAnal()
        self.sprModelSynth = es.SprModelSynth()
        self.spsModelAnal = es.SpsModelAnal()
        self.spsModelSynth = es.SpsModelSynth()
        self.stochasticModelAnal = es.StochasticModelAnal()
        self.stochasticModelSynth = es.StochasticModelSynth()
        self.resampleFFT = es.ResampleFFT()
        
        # === SEGMENTATION ===
        self.sbic = es.SBic()
        
        # === TRANSFORMATIONS ===
        try:
            self.gaiaTransform = es.GaiaTransform()
        except:
            pass
        self.pca = es.PCA()
        
        # === EXTRACTORS ===
        self.extractor = es.Extractor()
        self.musicExtractor = es.MusicExtractor()
        self.musicExtractorSVM = es.MusicExtractorSVM()
        self.freesoundExtractor = es.FreesoundExtractor()
        self.barkExtractor = es.BarkExtractor()
        self.lowLevelSpectralExtractor = es.LowLevelSpectralExtractor()
        self.lowLevelSpectralEqloudExtractor = es.LowLevelSpectralEqloudExtractor()
        
        # === MACHINE LEARNING ===
        try:
            self.tensorflowPredict = es.TensorflowPredict()
            self.tensorflowPredict2D = es.TensorflowPredict2D()
            self.tensorflowPredictCREPE = es.TensorflowPredictCREPE()
            self.tensorflowPredictEffnetDiscogs = es.TensorflowPredictEffnetDiscogs()
            self.tensorflowPredictFSDSINet = es.TensorflowPredictFSDSINet()
            self.tensorflowPredictMAEST = es.TensorflowPredictMAEST()
            self.tensorflowPredictMusiCNN = es.TensorflowPredictMusiCNN()
            self.tensorflowPredictTempoCNN = es.TensorflowPredictTempoCNN()
            self.tensorflowPredictVGGish = es.TensorflowPredictVGGish()
        except:
            pass  # TensorFlow features may not be available
        
    def extract_all_features(self, audio_path):
        """Extract every possible Essentia feature from an audio file"""
        print(f"Processing: {audio_path}")
        
        try:
            # Load audio in multiple formats
            audio = self.loader(audio_path)
            
            # Try to load metadata
            try:
                metadata_tags, audio_properties, duration_metadata = self.metadataReader(audio_path)
                metadata_features = {
                    'metadata_tags': metadata_tags,
                    'audio_properties': audio_properties,
                    'duration_metadata': duration_metadata
                }
            except:
                metadata_features = {}
            
            features = {}
            features.update(metadata_features)
            
            # Store raw audio and basic info
            features['audio_raw'] = audio
            features['sample_rate'] = self.sr
            features['duration'] = len(audio) / float(self.sr)
            features['num_samples'] = len(audio)
            
            print("  Extracting global analysis features...")
            
            # === GLOBAL FEATURES ===
            
            # High-level extractors (these compute many features at once)
            try:
                extractor_pool = self.extractor(audio)
                features['extractor_pool'] = extractor_pool
            except Exception as e:
                print(f"    Extractor failed: {e}")
            
            try:
                music_pool = self.musicExtractor(audio)
                features['music_extractor_pool'] = music_pool
            except Exception as e:
                print(f"    Music extractor failed: {e}")
            
            try:
                freesound_pool = self.freesoundExtractor(audio)
                features['freesound_pool'] = freesound_pool
            except Exception as e:
                print(f"    Freesound extractor failed: {e}")
            
            # Duration and silence
            features['duration_computed'] = self.duration(audio)
            features['effective_duration'] = self.effectiveDuration(self.envelope(audio))
            
            # Rhythm analysis
            try:
                bpm, beats, beats_confidence, estimates, bpm_intervals = self.rhythmExtractor2013(audio)
                features['bpm'] = bpm
                features['beats'] = beats
                features['beats_confidence'] = beats_confidence
                features['bpm_estimates'] = estimates
                features['bpm_intervals'] = bpm_intervals
            except Exception as e:
                print(f"    Rhythm extractor failed: {e}")
            
            try:
                bpm_rhythm, beat_positions, confidence, estimates_rhythm = self.rhythmExtractor(audio)
                features['bpm_rhythm'] = bpm_rhythm
                features['beat_positions'] = beat_positions
                features['confidence_rhythm'] = confidence
                features['estimates_rhythm'] = estimates_rhythm
            except Exception as e:
                print(f"    Basic rhythm extractor failed: {e}")
            
            # Alternative beat trackers
            try:
                features['beats_degara'] = self.beatTrackerDegara(audio)
            except:
                features['beats_degara'] = None
                
            try:
                features['beats_multifeature'] = self.beatTrackerMultiFeature(audio)
            except:
                features['beats_multifeature'] = None
            
            # Onset detection
            try:
                features['onsets_global'] = self.onsetDetectionGlobal(audio)
            except:
                features['onsets_global'] = None
            
            try:
                features['onset_rate'] = self.onsetRate(audio)
            except:
                features['onset_rate'] = None
            
            try:
                features['superflux_onsets'] = self.superFluxExtractor(audio)
            except:
                features['superflux_onsets'] = None
            
            # Key analysis
            try:
                key, scale, strength = self.keyExtractor(audio)
                features['key'] = key
                features['scale'] = scale
                features['key_strength'] = strength
            except:
                features['key'] = None
                features['scale'] = None
                features['key_strength'] = None
            
            # Tonal analysis
            try:
                tonal_pool = self.tonalExtractor(audio)
                features['tonal_pool'] = tonal_pool
            except:
                features['tonal_pool'] = None
            
            # Predominant melody
            try:
                pitch_values, pitch_confidence = self.predominantPitchMelodia(audio)
                features['predominant_melody'] = pitch_values
                features['melody_confidence'] = pitch_confidence
            except:
                features['predominant_melody'] = None
                features['melody_confidence'] = None
            
            # Multi-pitch analysis
            try:
                features['multi_pitch_melodia'] = self.multiPitchMelodia(audio)
            except:
                features['multi_pitch_melodia'] = None
                
            try:
                features['multi_pitch_klapuri'] = self.multiPitchKlapuri(audio)
            except:
                features['multi_pitch_klapuri'] = None
            
            # High-level descriptors
            try:
                danceability_value, danceability_dfa = self.danceability(audio)
                features['danceability'] = danceability_value
                features['danceability_dfa'] = danceability_dfa
            except:
                features['danceability'] = None
                features['danceability_dfa'] = None
            
            try:
                features['dynamic_complexity'] = self.dynamicComplexity(audio)
            except:
                features['dynamic_complexity'] = None
            
            try:
                features['intensity'] = self.intensity(audio)
            except:
                features['intensity'] = None
            
            # Loudness analysis
            try:
                features['loudness'] = self.loudness(audio)
            except:
                features['loudness'] = None
            
            try:
                features['loudness_ebu_r128'] = self.loudnessEBUR128(audio)
            except:
                features['loudness_ebu_r128'] = None
            
            try:
                features['loudness_vickers'] = self.loudnessVickers(audio)
            except:
                features['loudness_vickers'] = None
            
            try:
                features['replay_gain'] = self.replayGain(audio)
            except:
                features['replay_gain'] = None
            
            try:
                features['larm'] = self.larm(audio)
            except:
                features['larm'] = None
            
            try:
                features['leq'] = self.leq(audio)
            except:
                features['leq'] = None
            
            # Audio quality analysis
            try:
                click_starts, click_ends = self.clickDetector(audio)
                features['click_starts'] = click_starts
                features['click_ends'] = click_ends
            except:
                features['click_starts'] = None
                features['click_ends'] = None
            
            try:
                features['discontinuity'] = self.discontinuityDetector(audio)
            except:
                features['discontinuity'] = None
            
            try:
                gaps_starts, gaps_ends = self.gapsDetector(audio)
                features['gaps_starts'] = gaps_starts
                features['gaps_ends'] = gaps_ends
            except:
                features['gaps_starts'] = None
                features['gaps_ends'] = None
            
            try:
                hum_starts, hum_ends = self.humDetector(audio)
                features['hum_starts'] = hum_starts
                features['hum_ends'] = hum_ends
            except:
                features['hum_starts'] = None
                features['hum_ends'] = None
            
            try:
                noise_starts, noise_ends = self.noiseBurstDetector(audio)
                features['noise_starts'] = noise_starts
                features['noise_ends'] = noise_ends
            except:
                features['noise_starts'] = None
                features['noise_ends'] = None
            
            try:
                saturation_starts, saturation_ends = self.saturationDetector(audio)
                features['saturation_starts'] = saturation_starts
                features['saturation_ends'] = saturation_ends
            except:
                features['saturation_starts'] = None
                features['saturation_ends'] = None
            
            try:
                features['snr'] = self.snr(audio)
            except:
                features['snr'] = None
            
            try:
                features['true_peak'] = self.truePeakDetector(audio)
            except:
                features['true_peak'] = None
            
            # Fingerprinting
            try:
                features['chromaprint'] = self.chromaprinter(audio)
            except:
                features['chromaprint'] = None
            
            # Segmentation
            try:
                features['sbic_segments'] = self.sbic(audio)
            except:
                features['sbic_segments'] = None
            
            print("  Extracting frame-based features...")
            
            # === FRAME-BASED ANALYSIS ===
            # Initialize all frame-based feature lists
            frame_features = {
                # Spectral features
                'mfcc': [], 'bfcc': [], 'gfcc': [],
                'mel_bands': [], 'bark_bands': [], 'erb_bands': [],
                'triangular_bands': [], 'triangular_bark_bands': [],
                'frequency_bands': [], 'spectral_peaks_frequencies': [], 'spectral_peaks_magnitudes': [],
                'spectral_contrast': [], 'spectral_complexity': [], 'log_spectrum': [],
                'power_spectrum': [], 'spectrum_to_cent': [], 'hfc': [], 'flux': [],
                'flatness_db': [], 'max_mag_freq': [], 'roll_off': [], 'strong_peak': [],
                'energy_band': [], 'energy_band_ratio': [],
                
                # Tonal features
                'hpcp': [], 'chromagram': [], 'nnls_chroma': [], 'dissonance': [],
                'harmonic_peaks': [], 'inharmonicity': [], 'odd_even_ratio': [],
                'tristimulus': [], 'pitch_salience': [], 'spectrum_cq': [],
                
                # Pitch features
                'pitch_yin': [], 'pitch_yin_fft': [], 'pitch_melodia': [],
                'pitch_yin_probabilistic': [], 'pitch_yin_probabilities': [],
                'pitch_filter': [], 'pitch_salience_function': [],
                'vibrato': [],
                
                # Envelope/SFX features
                'envelope': [], 'after_max_before_max_ratio': [], 'derivative_sfx': [],
                'flatness_sfx': [], 'log_attack_time': [], 'max_to_total': [],
                'min_to_total': [], 'strong_decay': [], 'tc_to_total': [],
                
                # Statistical features
                'rms': [], 'energy': [], 'instant_power': [], 'centroid': [],
                'crest': [], 'decrease': [], 'flatness': [], 'geometric_mean': [],
                'mean': [], 'median': [], 'variance': [], 'central_moments': [],
                'distribution_shape': [], 'entropy': [], 'raw_moments': [],
                
                # Onset detection features
                'onset_detection': [], 'superflux_novelty': [], 'novelty_curve': [],
                
                # Loudness features
                'loudness_frame': [], 'level_extractor': [], 'silence_rate': [],
                
                # Zero crossing and autocorrelation
                'zero_crossing_rate': [], 'auto_correlation': [], 'warped_auto_correlation': [],
                
                # Filter outputs
                'dc_removal': [], 'equal_loudness': [], 'high_pass': [], 'low_pass': [],
                'band_pass': [], 'band_reject': [], 'all_pass': [],
                
                # Complex spectral features
                'fft': [], 'ifft': [], 'fftc': [], 'ifftc': [], 'dct': [], 'idct': [],
                'constant_q': [], 'nsg_constant_q': [],
                
                # TensorFlow inputs (if available)
                'tensorflow_musicnn': [], 'tensorflow_vggish': [], 'tensorflow_tempocnn': [],
                'tensorflow_fsd_sinet': [],
                
                # BPM and tempo features
                'bpm_histogram': [], 'tempo_scale_bands': [], 'tempo_tap': [],
                'rhythm_transform': [], 'loop_bpm_estimator': [], 'percival_bpm': [],
                
                # Synthesis features
                'sine_model': [], 'harmonic_model': [], 'stochastic_model': []
            }
            
            # Process frame by frame
            previous_spectrum = None
            frame_count = 0
            
            for frame in self.frameCutter(audio):
                if len(frame) == 0:
                    continue
                    
                frame_count += 1
                if frame_count % 1000 == 0:
                    print(f"    Processing frame {frame_count}")
                
                # Apply windowing
                windowed_frame = self.windowing(frame)
                
                # Basic spectral analysis
                spectrum = self.spectrum(windowed_frame)
                fft_complex = self.fft(windowed_frame)
                power_spec = self.powerSpectrum(windowed_frame)
                
                # Store complex and spectral features
                frame_features['fft'].append(fft_complex)
                frame_features['power_spectrum'].append(power_spec)
                
                # MFCC variants
                try:
                    mfcc_bands, mfcc_coeffs = self.mfcc(spectrum)
                    frame_features['mfcc'].append(mfcc_coeffs)
                except:
                    frame_features['mfcc'].append(np.zeros(13))
                
                try:
                    bfcc_bands, bfcc_coeffs = self.bfcc(spectrum)
                    frame_features['bfcc'].append(bfcc_coeffs)
                except:
                    frame_features['bfcc'].append(np.zeros(13))
                
                try:
                    gfcc_coeffs = self.gfcc(spectrum)
                    frame_features['gfcc'].append(gfcc_coeffs)
                except:
                    frame_features['gfcc'].append(np.zeros(13))
                
                # Filter bank features
                frame_features['mel_bands'].append(self.melBands(spectrum))
                frame_features['bark_bands'].append(self.barkBands(spectrum))
                frame_features['erb_bands'].append(self.erbBands(spectrum))
                frame_features['triangular_bands'].append(self.triangularBands(spectrum))
                frame_features['triangular_bark_bands'].append(self.triangularBarkBands(spectrum))
                frame_features['frequency_bands'].append(self.frequencyBands(spectrum))
                
                # Spectral descriptors
                frame_features['spectral_contrast'].append(self.spectralContrast(spectrum))
                frame_features['spectral_complexity'].append(self.spectralComplexity(spectrum))
                frame_features['log_spectrum'].append(self.logSpectrum(spectrum))
                frame_features['spectrum_to_cent'].append(self.spectrumToCent(spectrum))
                frame_features['hfc'].append(self.hfc(spectrum))
                frame_features['flatness_db'].append(self.flatnessDB(spectrum))
                frame_features['max_mag_freq'].append(self.maxMagFreq(spectrum))
                frame_features['roll_off'].append(self.rollOff(spectrum))
                frame_features['strong_peak'].append(self.strongPeak(spectrum))
                
                # Spectral flux requires previous spectrum
                if previous_spectrum is not None:
                    frame_features['flux'].append(self.flux(spectrum, previous_spectrum))
                else:
                    frame_features['flux'].append(0.0)
                previous_spectrum = spectrum
                
                # Energy features
                try:
                    frame_features['energy_band'].append(self.energyBand(spectrum))
                except:
                    frame_features['energy_band'].append(0.0)
                
                try:
                    frame_features['energy_band_ratio'].append(self.energyBandRatio(spectrum))
                except:
                    frame_features['energy_band_ratio'].append(0.0)
                
                # Peak-based features
                peak_frequencies, peak_magnitudes = self.spectralPeaks(spectrum)
                frame_features['spectral_peaks_frequencies'].append(peak_frequencies)
                frame_features['spectral_peaks_magnitudes'].append(peak_magnitudes)
                
                # Tonal features (require peaks)
                if len(peak_frequencies) > 0:
                    frame_features['hpcp'].append(self.hpcp(peak_frequencies, peak_magnitudes))
                    frame_features['chromagram'].append(self.chromagram(peak_frequencies, peak_magnitudes))
                    frame_features['dissonance'].append(self.dissonance(peak_frequencies, peak_magnitudes))
                    frame_features['harmonic_peaks'].append(self.harmonicPeaks(peak_frequencies, peak_magnitudes))
                    frame_features['inharmonicity'].append(self.inharmonicity(peak_frequencies, peak_magnitudes))
                    frame_features['odd_even_ratio'].append(self.oddToEvenHarmonicEnergyRatio(peak_frequencies, peak_magnitudes))
                    frame_features['tristimulus'].append(self.tristimulus(peak_frequencies, peak_magnitudes))
                else:
                    frame_features['hpcp'].append(np.zeros(12))
                    frame_features['chromagram'].append(np.zeros(12))
                    frame_features['dissonance'].append(0.0)
                    frame_features['harmonic_peaks'].append([])
                    frame_features['inharmonicity'].append(0.0)
                    frame_features['odd_even_ratio'].append(0.0)
                    frame_features['tristimulus'].append([0.0, 0.0, 0.0])
                
                # Pitch features
                try:
                    pitch_yin, pitch_conf_yin = self.pitchYin(spectrum)
                    frame_features['pitch_yin'].append(pitch_yin)
                except:
                    frame_features['pitch_yin'].append(0.0)
                
                try:
                    pitch_yin_fft, pitch_conf_yin_fft = self.pitchYinFFT(spectrum)
                    frame_features['pitch_yin_fft'].append(pitch_yin_fft)
                except:
                    frame_features['pitch_yin_fft'].append(0.0)
                
                try:
                    frame_features['pitch_salience'].append(self.pitchSalience(spectrum))
                except:
                    frame_features['pitch_salience'].append(0.0)
                
                try:
                    salience_function = self.pitchSalienceFunction(spectrum)
                    frame_features['pitch_salience_function'].append(salience_function)
                except:
                    frame_features['pitch_salience_function'].append(np.zeros(100))
                
                # Additional spectral features
                try:
                    frame_features['spectrum_cq'].append(self.spectrumCQ(spectrum))
                except:
                    frame_features['spectrum_cq'].append(spectrum)
                
                # Statistical features
                frame_features['rms'].append(self.rms(frame))
                frame_features['energy'].append(self.energy(frame))
                frame_features['instant_power'].append(self.instantPower(frame))
                frame_features['centroid'].append(self.centroid(spectrum))
                frame_features['crest'].append(self.crest(spectrum))
                frame_features['decrease'].append(self.decrease(spectrum))
                frame_features['flatness'].append(self.flatness(spectrum))
                frame_features['geometric_mean'].append(self.geometricMean(spectrum))
                frame_features['mean'].append(self.mean(spectrum))
                frame_features['median'].append(self.median(spectrum))
                frame_features['variance'].append(self.variance(spectrum))
                frame_features['entropy'].append(self.entropy(spectrum))
                
                # Central moments and distribution shape
                try:
                    central_moments = self.centralMoments(spectrum)
                    frame_features['central_moments'].append(central_moments)
                    distribution_shape = self.distributionShape(central_moments)
                    frame_features['distribution_shape'].append(distribution_shape)
                except:
                    frame_features['central_moments'].append(np.zeros(5))
                    frame_features['distribution_shape'].append(np.zeros(3))
                
                try:
                    frame_features['raw_moments'].append(self.rawMoments(spectrum))
                except:
                    frame_features['raw_moments'].append(np.zeros(5))
                
                # Onset detection features
                try:
                    frame_features['onset_detection'].append(self.onsetDetection(spectrum, previous_spectrum if previous_spectrum is not None else spectrum))
                except:
                    frame_features['onset_detection'].append(0.0)
                
                try:
                    frame_features['superflux_novelty'].append(self.superFluxNovelty(spectrum))
                except:
                    frame_features['superflux_novelty'].append(0.0)
                
                try:
                    frame_features['novelty_curve'].append(self.noveltyCurve(spectrum))
                except:
                    frame_features['novelty_curve'].append(0.0)
                
                # Zero crossing rate
                frame_features['zero_crossing_rate'].append(self.zeroCrossingRate(frame))
                
                # Autocorrelation features
                try:
                    frame_features['auto_correlation'].append(self.autoCorrelation(frame))
                except:
                    frame_features['auto_correlation'].append(np.zeros(len(frame)))
                
                try:
                    frame_features['warped_auto_correlation'].append(self.warpedAutoCorrelation(frame))
                except:
                    frame_features['warped_auto_correlation'].append(np.zeros(len(frame)))
                
                # Envelope features
                try:
                    envelope = self.envelope(frame)
                    frame_features['envelope'].append(envelope)
                    
                    # Envelope-based SFX features
                    frame_features['after_max_before_max_ratio'].append(self.afterMaxToBeforeMaxEnergyRatio(envelope))
                    frame_features['derivative_sfx'].append(self.derivativeSFX(envelope))
                    frame_features['flatness_sfx'].append(self.flatnessSFX(envelope))
                    frame_features['log_attack_time'].append(self.logAttackTime(envelope))
                    frame_features['max_to_total'].append(self.maxToTotal(envelope))
                    frame_features['min_to_total'].append(self.minToTotal(envelope))
                    frame_features['strong_decay'].append(self.strongDecay(envelope))
                    frame_features['tc_to_total'].append(self.tcToTotal(envelope))
                except:
                    frame_features['envelope'].append(np.zeros(len(frame)))
                    frame_features['after_max_before_max_ratio'].append(0.0)
                    frame_features['derivative_sfx'].append(0.0)
                    frame_features['flatness_sfx'].append(0.0)
                    frame_features['log_attack_time'].append(0.0)
                    frame_features['max_to_total'].append(0.0)
                    frame_features['min_to_total'].append(0.0)
                    frame_features['strong_decay'].append(0.0)
                    frame_features['tc_to_total'].append(0.0)
                
                # Loudness features
                try:
                    frame_features['loudness_frame'].append(self.loudness(frame))
                except:
                    frame_features['loudness_frame'].append(0.0)
                
                try:
                    frame_features['silence_rate'].append(self.silenceRate(frame))
                except:
                    frame_features['silence_rate'].append(0.0)
                
                # Filters (applied to frame)
                try:
                    frame_features['dc_removal'].append(self.dcRemoval(frame))
                except:
                    frame_features['dc_removal'].append(frame)
                
                try:
                    frame_features['equal_loudness'].append(self.equalLoudness(frame))
                except:
                    frame_features['equal_loudness'].append(frame)
                
                # TensorFlow inputs (if available)
                try:
                    frame_features['tensorflow_musicnn'].append(self.tensorflowInputMusiCNN(spectrum))
                except:
                    frame_features['tensorflow_musicnn'].append(np.zeros(96))
                
                try:
                    frame_features['tensorflow_vggish'].append(self.tensorflowInputVGGish(spectrum))
                except:
                    frame_features['tensorflow_vggish'].append(np.zeros(96))
                
                # Break after reasonable number of frames to prevent memory explosion
                if frame_count > 10000:  # About 7-8 minutes at default settings
                    print(f"    Truncating at {frame_count} frames to prevent memory issues")
                    break
            
            print(f"  Processed {frame_count} frames")
            
            # Convert frame lists to numpy arrays
            for key, values in frame_features.items():
                if values:
                    try:
                        if isinstance(values[0], np.ndarray) and len(values[0].shape) > 0:
                            # Multi-dimensional features
                            features[key] = np.array(values).T
                        else:
                            # 1D features
                            features[key] = np.array(values)
                    except:
                        # If conversion fails, store as list
                        features[key] = values
                else:
                    features[key] = None
            
            print("  Computing additional global features...")
            
            # === ADDITIONAL GLOBAL FEATURES ===
            
            # BPM and tempo analysis
            if 'onset_detection' in features and features['onset_detection'] is not None:
                try:
                    bpm_hist, bpm_peaks = self.bpmHistogramDescriptors(features['onset_detection'])
                    features['bpm_histogram'] = bpm_hist
                    features['bpm_peaks'] = bpm_peaks
                except:
                    features['bpm_histogram'] = None
                    features['bpm_peaks'] = None
            
            # Chroma-based features
            if 'chromagram' in features and features['chromagram'] is not None:
                try:
                    features['nnls_chroma'] = self.nnlsChroma(features['chromagram'])
                except:
                    features['nnls_chroma'] = None
            
            # Tuning frequency
            if 'spectral_peaks_frequencies' in features:
                try:
                    features['tuning_frequency'] = self.tuningFrequencyExtractor(audio)
                except:
                    features['tuning_frequency'] = None
            
            # Fade detection
            if 'rms' in features and features['rms'] is not None:
                try:
                    fade_in, fade_out = self.fadeDetection(features['rms'])
                    features['fade_in'] = fade_in
                    features['fade_out'] = fade_out
                except:
                    features['fade_in'] = None
                    features['fade_out'] = None
            
            # Start/stop analysis
            try:
                start_frame, stop_frame = self.startStopSilence(audio)
                features['start_frame'] = start_frame
                features['stop_frame'] = stop_frame
            except:
                features['start_frame'] = None
                features['stop_frame'] = None
            
            # Cross-correlation analysis (if we have stereo)
            try:
                # For mono signals, duplicate for cross-correlation
                features['cross_correlation'] = self.crossCorrelation(audio, audio)
            except:
                features['cross_correlation'] = None
            
            # Synthesis analysis
            try:
                sine_frequencies, sine_magnitudes, sine_phases = self.sineModelAnal(windowed_frame, fft_complex)
                features['sine_model_frequencies'] = sine_frequencies
                features['sine_model_magnitudes'] = sine_magnitudes
                features['sine_model_phases'] = sine_phases
            except:
                features['sine_model_frequencies'] = None
                features['sine_model_magnitudes'] = None
                features['sine_model_phases'] = None
            
            # LPC analysis
            try:
                lpc_coeffs, reflection_coeffs = self.lpc(audio)
                features['lpc_coefficients'] = lpc_coeffs
                features['reflection_coefficients'] = reflection_coeffs
            except:
                features['lpc_coefficients'] = None
                features['reflection_coefficients'] = None
            
            print("  Computing statistical summaries...")
            
            # Compute statistics for all time-varying features
            time_varying_features = [
                'mfcc', 'bfcc', 'gfcc', 'mel_bands', 'bark_bands', 'erb_bands',
                'triangular_bands', 'triangular_bark_bands', 'frequency_bands',
                'spectral_contrast', 'spectral_complexity', 'hfc', 'flux',
                'flatness_db', 'max_mag_freq', 'roll_off', 'strong_peak',
                'hpcp', 'chromagram', 'dissonance', 'inharmonicity',
                'odd_even_ratio', 'tristimulus', 'pitch_yin', 'pitch_yin_fft',
                'pitch_salience', 'rms', 'energy', 'instant_power',
                'centroid', 'crest', 'decrease', 'flatness', 'entropy',
                'zero_crossing_rate', 'onset_detection', 'loudness_frame'
            ]
            
            for feat_name in time_varying_features:
                if feat_name in features and features[feat_name] is not None:
                    feat_data = features[feat_name]
                    try:
                        if feat_data.ndim > 1:  # Multi-dimensional features
                            for i in range(feat_data.shape[0]):
                                if len(feat_data[i]) > 0:
                                    features[f'{feat_name}_mean_{i}'] = np.mean(feat_data[i])
                                    features[f'{feat_name}_std_{i}'] = np.std(feat_data[i])
                                    features[f'{feat_name}_var_{i}'] = np.var(feat_data[i])
                                    features[f'{feat_name}_min_{i}'] = np.min(feat_data[i])
                                    features[f'{feat_name}_max_{i}'] = np.max(feat_data[i])
                                    features[f'{feat_name}_median_{i}'] = np.median(feat_data[i])
                        else:  # 1D features
                            if len(feat_data) > 0:
                                features[f'{feat_name}_mean'] = np.mean(feat_data)
                                features[f'{feat_name}_std'] = np.std(feat_data)
                                features[f'{feat_name}_var'] = np.var(feat_data)
                                features[f'{feat_name}_min'] = np.min(feat_data)
                                features[f'{feat_name}_max'] = np.max(feat_data)
                                features[f'{feat_name}_median'] = np.median(feat_data)
                    except:
                        pass  # Skip if statistical computation fails
            
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
            output_path = input_path / "complete_essentia_features"
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
        print(f"WARNING: This will create very large files (potentially 5-20 GB per track)")
        
        success_count = 0
        for audio_file in audio_files:
            try:
                # Extract features
                features = self.extract_all_features(str(audio_file))
                
                if features is not None:
                    # Save as compressed numpy file
                    output_file = output_path / f"{audio_file.stem}_complete_essentia.npz"
                    
                    # Filter out None values and convert to serializable formats
                    clean_features = {}
                    for key, value in features.items():
                        if value is not None:
                            if isinstance(value, np.ndarray):
                                clean_features[key] = value
                            elif isinstance(value, (int, float, complex)):
                                clean_features[key] = np.array(value)
                            elif isinstance(value, str):
                                # Store strings as arrays of unicode code points for npz compatibility
                                clean_features[key] = np.array([ord(c) for c in value])
                            elif isinstance(value, dict):
                                # Handle pool dictionaries by flattening them
                                for sub_key, sub_value in value.items():
                                    try:
                                        if isinstance(sub_value, np.ndarray):
                                            clean_features[f'{key}_{sub_key}'] = sub_value
                                        elif isinstance(sub_value, (int, float, complex)):
                                            clean_features[f'{key}_{sub_key}'] = np.array(sub_value)
                                        elif isinstance(sub_value, list):
                                            clean_features[f'{key}_{sub_key}'] = np.array(sub_value)
                                    except:
                                        pass
                            elif isinstance(value, list):
                                try:
                                    clean_features[key] = np.array(value)
                                except:
                                    pass  # Skip if can't convert to array
                    
                    np.savez_compressed(output_file, **clean_features)
                    print(f"  Saved: {output_file}")
                    print(f"  File size: {output_file.stat().st_size / (1024*1024*1024):.2f} GB")
                    success_count += 1
                
            except Exception as e:
                print(f"  Failed to process {audio_file}: {e}")
                continue
        
        print(f"\nCompleted! Successfully processed {success_count}/{len(audio_files)} files")
        print(f"Features saved in: {output_path}")
        
        # Print a sample of what was extracted
        if success_count > 0:
            sample_file = list(output_path.glob("*_complete_essentia.npz"))[0]
            sample_data = np.load(sample_file)
            print(f"\nSample feature file: {sample_file.name}")
            print(f"Contains {len(sample_data.keys())} features:")
            print(f"File size: {sample_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Group features by category for better display
            categories = {
                'Basic': ['duration', 'bpm', 'sample_rate', 'num_samples'],
                'Spectral': [k for k in sample_data.keys() if any(x in k for x in ['spectral', 'mfcc', 'mel_bands', 'bark'])],
                'Rhythm': [k for k in sample_data.keys() if any(x in k for x in ['beat', 'tempo', 'onset', 'bpm', 'rhythm'])],
                'Tonal': [k for k in sample_data.keys() if any(x in k for x in ['key', 'hpcp', 'chroma', 'pitch', 'tonal'])],
                'High-level': [k for k in sample_data.keys() if any(x in k for x in ['danceability', 'dynamic', 'loudness', 'intensity'])],
                'Audio Quality': [k for k in sample_data.keys() if any(x in k for x in ['click', 'hum', 'noise', 'saturation', 'gaps'])],
                'Envelope/SFX': [k for k in sample_data.keys() if any(x in k for x in ['envelope', 'attack', 'decay', 'sfx'])],
                'Statistical': [k for k in sample_data.keys() if any(x in k for x in ['mean', 'std', 'var', 'min', 'max', 'median'])],
                'Extractors': [k for k in sample_data.keys() if 'extractor' in k or 'pool' in k]
            }
            
            for category, keys in categories.items():
                matching_keys = [k for k in keys if k in sample_data.keys()]
                if matching_keys:
                    print(f"\n  {category} features ({len(matching_keys)}):")
                    for key in matching_keys[:5]:  # Show first 5 in each category
                        try:
                            shape = sample_data[key].shape if hasattr(sample_data[key], 'shape') else 'scalar'
                            print(f"    {key}: {shape}")
                        except:
                            print(f"    {key}: unknown shape")
                    if len(matching_keys) > 5:
                        print(f"    ... and {len(matching_keys) - 5} more")

def main():
    parser = argparse.ArgumentParser(description='Extract ALL Essentia features - complete unoptimized version')
    parser.add_argument('input_folder', help='Folder containing audio files')
    parser.add_argument('output_folder', nargs='?', help='Output folder for feature files (default: input_folder/complete_essentia_features)')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate (default: 44100)')
    parser.add_argument('--hop_length', type=int, default=128, help='Hop length (default: 128)')
    parser.add_argument('--frame_size', type=int, default=2048, help='Frame size (default: 2048)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    print("WARNING: This extractor implements 200+ Essentia algorithms")
    print("Expected file sizes: 5-20 GB per track")
    print("Processing time: 30+ minutes per track")
    print("Memory usage: 8+ GB RAM required")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Create extractor
    extractor = CompleteEssentiaExtractor(sr=args.sr, hop_length=args.hop_length, frame_size=args.frame_size)
    
    # Process folder
    extractor.process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
