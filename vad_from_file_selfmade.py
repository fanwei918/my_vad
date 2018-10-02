#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:58:59 2018

@author: weif
"""



import os
import numpy as np
import glob


import matplotlib.pyplot as plt

from scipy.io import wavfile


import struct


import soundfile as sf
import librosa

import wesserstein

import webrtcvad


# get the zero-crossing-rate for each frame within the signal, append and return one array
def get_zcr_by_frame(signal, framelen_t=0.02, sr=16000):
    framelen = int(framelen_t * sr)
    n_frames = signal.shape[0] // framelen
    zcr_array = np.zeros((n_frames,))
    for n in range(n_frames):
        sub_signal = signal[n * framelen:(n + 1) * framelen]
        zcr_array[n] = (np.sign(sub_signal) != np.sign(np.roll(sub_signal, 1))).sum() * 1.0 / sub_signal.shape[0]

    return zcr_array

# get the short_time_energy for each frame within the signal, append and return one array
def get_ste_by_frame(signal, framelen_t=0.02, sr=16000, scale=1):
    framelen = int(framelen_t * sr)
    n_frames = signal.shape[0] // framelen
    ste_array = np.zeros((n_frames,))
    for n in range(n_frames):
        sub_signal = signal[n * framelen:(n + 1) * framelen]
        ste_array[n] = (sub_signal ** 2).sum() / scale ** 2 / sub_signal.shape[0]

    return ste_array

def get_ste_by_frame_simple(signal, framelen_t=0.5, sr=16000, scale=1):
    framelen = int(framelen_t * sr)
    n_frames = signal.shape[0] // framelen
    ste_array = np.zeros((n_frames,))
    for n in range(n_frames):
        sub_signal = signal[n * framelen:(n + 1) * framelen]
        ste_array[n] = np.abs(sub_signal).mean()
    return ste_array


def get_event_onset(ste, framelen_t=0.02):
    egy_floor = ste.mean()/100
    onset = np.where(np.logical_and(ste>0 , ste > egy_floor) )[0]
    

    start  = onset[0]
    stop = np.minimum(onset[-1] + int(0.5/framelen_t) , ste.shape[0])
    
    return start, stop


# HZCRR and LSTER seems not quite good if want to distinguish voice and general sounds.
# try these criteries:
#    1. <3kHz > 90%
#    2. no single tone
#    3. have considerable spectrum variance with in one onset event.
    
def get_binned_spectrogram(y, nfft=1024, window=False, hop=1024, n_bins=128, option_abs_sqr = None):
        """
        important params:
            n_bins: if None, returns original spectrogram, has highest resolution in frequency domain (== nfft/2)
                    if not None (must be smaller than nfft/2) it will sum the adjancent frequency response into 1 bin, to get binned freq response.
            window / hop :  usually used in a combined manner to get:
                    window =0, hop = nfft : No windowing operation
                    window = 'han', hop = nfft/2 : Hanning window, most commonly applied windowing technique in signal processing.
            option_abs_sqr:  if None, use the abs magnitude of frequency response. if 'sqr', use the square of the magnitude of frequency response.
        """
        if window:
            D = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop, center=False))
        else:
            D = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop, center=False, window=np.ones((nfft,))))
            
        if option_abs_sqr == 'sqr':
            D = D**2

        if n_bins is None:
            return D
        else:
            no_freq_in_bin = nfft // 2 // n_bins
            binned_mag = np.zeros((n_bins, D.shape[1]))
            for n in range(n_bins):
                binned_mag[n, :] = D[n * no_freq_in_bin: (n + 1) * no_freq_in_bin, :].mean(axis=0)
            return binned_mag



def Half_peak_width(pos_def_arr, power = 2):
    '''
    consider an positive definite array as distribution
    find the half-peak width of that distribution 
    '''
    half_max = pos_def_arr.max() / (2**power)
    cnt = (pos_def_arr < half_max).sum()
    return cnt/pos_def_arr.shape[0]


def vad_condition1(spectrogram, sr = 16000):
    
    n_frames = spectrogram.shape[1]
    
    band_cut0 = int( (1.0*spectrogram.shape[0])/(sr/2)*3000)
    band_cut1 = int( (1.0*spectrogram.shape[0])/(sr/2)*1500 )
    band_cut2 = int( (1.0*spectrogram.shape[0])/(sr/2)*750 )
    
    band_flatness = np.zeros((n_frames,))
    band_flatness1 = np.zeros((n_frames,))
    band_below_ratio1 = np.zeros((n_frames,))
    band_below_ratio2 = np.zeros((n_frames,))
    egy_per_frame = np.zeros((n_frames,))
    
    egy_focus = np.zeros((n_frames,))
    
    
    for n in range(n_frames):
        sum_egy = spectrogram[:,n].sum()+0.00001
        
        band_flatness1[n] =  spectrogram[:band_cut0,n].max() /( spectrogram[:band_cut0,n].mean()+ 0.00001)  #gmean( spectrogram[:band_cut0,n] ) /  ( spectrogram[:band_cut0,n].mean()+ 0.00001)
        
        max_band = np.argmax(spectrogram[:band_cut0,n])
        nbr_max_band = np.arange( max(0,max_band -5) , max_band+5)
        band_flatness[n] =  spectrogram[:band_cut0,n].max() /( spectrogram[nbr_max_band,n].mean()+ 0.00001)  #gmean( spectrogram[:band_cut0,n] ) /  ( spectrogram[:band_cut0,n].mean()+ 0.00001)

        band_below_ratio1[n] = spectrogram[:band_cut1,n].sum() / sum_egy
        band_below_ratio2[n] = spectrogram[:band_cut2,n].sum() / sum_egy
        egy_per_frame[n]= sum_egy
        
        egy_focus[n] = Half_peak_width(spectrogram[:,n])
             
    return band_flatness, band_flatness1, band_below_ratio1, band_below_ratio2, egy_per_frame, egy_focus





def spectro_simi_measure(spectrogram):
    
    n_frames = spectrogram.shape[1]
    
    spectro_similarity = np.zeros((n_frames-1,))
    spectro_similarity_wt = np.zeros((n_frames-1,))
    
    for n in range(n_frames-1):
#        spectro_similarity[n] = np.corrcoef(spectrogram[:,n]+1,spectrogram[:,n+1]+1 )[0,1]
        spectro_similarity[n]= wesserstein.wesserstein_distance_1d(spectrogram[:,n]+1,spectrogram[:,n+1]+1)
        spectro_similarity_wt[n] = np.minimum( spectrogram[:,n].sum(), spectrogram[:,n+1].sum() )
        
    return spectro_similarity[ np.isnan(spectro_similarity)==False ], spectro_similarity_wt[ np.isnan(spectro_similarity)==False]
        

#%%



no_clips = 0
vad_detected1 = []
vad_detected2 = []
no_vad_detected = 0



feature_array = []


#for filename in glob.glob('../data_from_device/device_0.wav'):
#for filename in glob.glob('../live_data/*.wav'):
#for filename in glob.glob('../Event_detection/data/outdoor_small_set/1st*/*/*.wav'):
#for filename in glob.glob('../Event_detection/data/outdoor_small_set/918*/*.wav'):
#for filename in glob.glob('../Event_detection/data/human_command/*/*.wav'):
#for filename in glob.glob('../Event_detection/data/outdoor_small_set/*/*.ogg'):

#for filename in glob.glob('../data_from_device/*.wav'):

for filename in glob.glob('/Users/weif/Documents/Occupancy/Event_detection/snd_effcts/human_remark_talk/*.wav'):
    
    
    if filename.split('.')[-1] == 'wav':
        sample_rate, samples = wavfile.read(filename)
    elif filename.split('.')[-1] == 'ogg':
        samples,sample_rate = sf.read(filename, dtype = 'float32')
    
    
    if samples.ndim>1 and samples.shape[1]>1:
        samples=samples[:,0]
        
    if sample_rate!=16000:
        samples = (librosa.resample(samples,sample_rate,16000)).astype(float)
        samples = samples/(np.abs(samples).max()+0.00001) *32768
        
    if np.abs(samples).max()<=1:
        samples *= 32768 
        
    if samples.shape[0]<80000:
        a = (np.random.random((80000,))-0.5)* 10
        place = int(np.random.random() * (80000-samples.shape[0]))
        a[place:place+samples.shape[0]]+=samples
        samples=a.copy()
    
    samples = np.int16(samples)
    no_clips+=1
    
    sample_rate = 16000
    
    spectrogram  =  get_binned_spectrogram(samples.astype(float))
#    feature_array.append(spectrogram)   
    

    band_flatness,band_flatness1, band_below_ratio1 ,band_below_ratio2, egy_per_frame , egy_focus = vad_condition1(spectrogram, sr = 16000)
    
    spectro_similarity, spectro_similarity_wt = spectro_simi_measure(spectrogram)
    
    feat1 = np.average(band_flatness,weights = egy_per_frame) 
    feat1_1 = np.average(band_flatness1,weights = egy_per_frame) 
    feat2 = np.average(band_below_ratio1,weights = egy_per_frame)
    feat3 = np.average(band_below_ratio2,weights = egy_per_frame)
    feat4 = np.average(spectro_similarity,weights = spectro_similarity_wt)
    
    feat6 = np.average(egy_focus,weights = egy_per_frame) 
    
    
    cond1 = feat1>2 and feat1<4
    cond1_1 = feat1_1 > 5 and feat1_1 < 13
    cond2 = feat2 > 0.38
    cond3 = feat3 > 0.18
    cond4 = feat4 > 2.5
    cond6 = feat6 > 0.5
    
    
    
    vad1 = webrtcvad.Vad()
    vad1.set_mode(3)
    
    samples = np.int16(samples)
    raw_samples = struct.pack("%dh" % len(samples), *samples)
    window_duration = 0.01 # duration in seconds
    samples_per_window = int(window_duration * sample_rate + 0.5)
    bytes_per_sample = 2
    
    segments = []
    speech_per_frame = []

    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
        
        if stop-start< 0.01*sample_rate:
            continue
        
        is_speech = vad1.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
                                  sample_rate = sample_rate)
    
        segments.append(dict(
           start = start,
           stop = stop,
           is_speech = is_speech))
        
        speech_per_frame.append(is_speech)

    detected1 = np.array(speech_per_frame).mean()
    cond5 = detected1>0.2
                

                
    
#    feature_array.append(np.array([feat1, feat1_1, feat2, feat3, feat4]))
    
#    break

    
    if cond1 and cond1_1 and cond2 and cond3 and cond4 and cond5 and cond6:
        no_vad_detected += 1
        
        
    
#    if no_clips%1000 == 999:
print( no_vad_detected, no_clips)

#%%

#    break
#
#feature_array = np.array(feature_array)
#print(no_clips)
#print(no_vad_detected)
#
#plt.hist(feature_array[:,0],40)
#plt.show()
#
#plt.hist(feature_array[:,1],40)
#plt.show()

#feature_vec=[]
#for n in range (64720):
#    
#    spectrogram = feature_array[n,:,:]
#    
#    band_flatness, band_flatness1, band_below_ratio1 ,band_below_ratio2, egy_per_frame , egy_focus = vad_condition1(spectrogram, sr = 16000)
#    
#    feat5 = np.average(egy_focus,weights = egy_per_frame) 
#    
#    feature_vec.append(feat5)
#    
#    
    
    