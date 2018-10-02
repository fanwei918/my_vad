#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:36:27 2018

@author: weif
"""

import matplotlib
#matplotlib.use('TkAgg')

import numpy as np
import struct
import pyaudio
import matplotlib.pyplot as plt
import librosa

#from matplotlib import animation

from scipy.io import wavfile


import webrtcvad

import datetime 
import wesserstein

from scipy.stats.mstats import gmean
#%%


CHUNK = 4410  # num of data samples read per 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

RECORD_SECONDS = 5
PREDICTION_SECONDS = 5


BLOCK = RECORD_SECONDS * RATE  # BLOCK is the prediction window
GAP = (PREDICTION_SECONDS * RATE) / CHUNK  # GAP should be int; it is the prediction gap
i = 0

def get_snd_max(signal, framelen_t=0.02, sr=16000):
        framelen = int(framelen_t * sr)
        n_frames = signal.shape[0] // framelen
        snd_max = np.zeros((n_frames,))
        for n in range(n_frames):
            snd_max[n] = signal[n * framelen:(n + 1) * framelen].max()

        return snd_max


def learn_noisefloor(y, framelen_t=0.02, sr=16000, alpha=0.5, inital_noise_floor=100):
        """
        y is a 1d time serie read from .wav file
        first split the serie into small frames (by framelen).
        in each frame we get what is the maximum signal magnitude, use this to build another time serie, whose size is y.shape[0]//framelen
        then we get the approximate noisefloor by low-evelope extraction.
        """
        framelen = int(framelen_t * sr)
        signal1 = y.copy()
        n_frames = signal1.shape[0] // framelen
        n_frames = n_frames
        signal1.resize(n_frames * framelen)
        signal1 = signal1.reshape((n_frames, framelen))
        max_mag_per_frame = np.abs(signal1).max(axis=1)

        noisefloor_local = np.zeros_like(max_mag_per_frame)
        noisefloor_local[0] = inital_noise_floor
        for n in range(1, max_mag_per_frame.shape[0]):
            if max_mag_per_frame[n] < noisefloor_local[n - 1]:
                noisefloor_local[n] = (1 - alpha) * noisefloor_local[n - 1] + alpha * max_mag_per_frame[n]
            else:
                noisefloor_local[n] = noisefloor_local[n - 1] + 1
        
        return noisefloor_local
    

def audio_event(snd_max, noise_floor, threshold=5, framelen_t=0.02, sr=16000, threshold_raw = 200):

        max_silence_gap = 20  # no. of frames. the maximum of silent gap within event. if reached, will split into 2
        # events. the corresponding time length is max_continuous_silence*framelen/samplingrate seconds.
        min_len_as_event = 6  # no of frames. the minimum consecutive length that the audio last which gets considered
        #  as event.
        tail_len = 10  # unit is no of frames. the allowed length of the tail for each event detected.

        def get_start_stop_pos(a):
            start = []
            stop = []
            if a[0] == 1:
                start.append(0)
            for n in range(1, a.shape[0]):
                if a[n - 1] == 0 and a[n] == 1:
                    start.append(n)
                elif a[n - 1] == 1 and a[n] == 0:
                    stop.append(n)
            if a[-1] == 1:
                stop.append(a.shape[0] - 1)
            return np.vstack([np.array(start), np.array(stop)]).transpose()

        n_frames = snd_max.shape[0]

        frame_onset = np.zeros((n_frames))

        for n in range(n_frames):
            if  ( snd_max[n] > noise_floor[n] * threshold ) and snd_max[n]>threshold_raw:
                frame_onset[n] = 1

        # go through the frame_onset, and merge the frames with shot gap;
        start_stop_pos = get_start_stop_pos(frame_onset)
        if start_stop_pos.shape[0] > 1:
            for n in range(1, start_stop_pos.shape[1]):
                if start_stop_pos[n, 0] - start_stop_pos[n - 1, 1] < max_silence_gap:
                    frame_onset[start_stop_pos[n - 1, 1]: start_stop_pos[n, 0]] = 1

        # go thruogh and abandon the events with < min_len_as_events. too short to be an event
        start_stop_pos = get_start_stop_pos(frame_onset)
        for n in range(start_stop_pos.shape[0]):
            if start_stop_pos[n, 1] - start_stop_pos[n, 0] < min_len_as_event:
                frame_onset[start_stop_pos[n, 0]: start_stop_pos[n, 1]] = 0

        # go through and add tails for each single events.
        start_stop_pos = get_start_stop_pos(frame_onset)
        for n in range(start_stop_pos.shape[0]):
            tail = np.minimum((start_stop_pos[n, 1] - start_stop_pos[n, 0]) / 2, tail_len)
            start_stop_pos[n, 1] += tail
            start_stop_pos[n, 1] = np.minimum(start_stop_pos[n, 1], frame_onset.shape[0])
            frame_onset[start_stop_pos[n, 0]: start_stop_pos[n, 1]] = 1

        event_onset = np.repeat(frame_onset,int(framelen_t*sr))
        
        return frame_onset, event_onset
    

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


NAME_REC = 'room1_'


noise_floor = np.array([100])


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

'''
initializ
'''

infrasnd = []

vad1 = webrtcvad.Vad()
# set aggressiveness from 0 to 3
vad1.set_mode(2)
sample_rate =16000

# Create the audio stream


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize the data container

frames = bytes(str([]), 'utf-16')

#process_len = 512

'''
Starting recording
'''

print("* recording")


try:
    while True:    
        
        
        record = stream.read(CHUNK, exception_on_overflow=False)
        frames = frames + record
        
        i = i + 1
        if i % GAP == 1 and i > BLOCK/CHUNK:
            frame_len = len(frames)
            if frame_len > (BLOCK * 2):  #the bytes type has 2 times length of the real data it representes
                frames = frames[frame_len - BLOCK*2:frame_len]
                data_form = "%dh" % (BLOCK)
                raw = (np.array(struct.unpack(data_form, frames))+0.5) / ((0x7FFF + 0.5))
                
                raw= (librosa.resample(raw,RATE,16000)*32768).astype(int)
                
                snd_max = get_snd_max(raw)
                noise_floor = learn_noisefloor(raw, inital_noise_floor = noise_floor[-1])
                frame_tag, event_tag = audio_event(snd_max, noise_floor =noise_floor )
                if event_tag.shape[0]<raw.shape[0]:
                    event_tag = np.pad(event_tag,(0,1),'constant', constant_values = raw.shape[0] - event_tag.shape[0])
                    
                
#                plt.plot(np.abs(np.fft.fft(raw.astype(float)))[1:21])
#                plt.show()
                    
#                infrasnd.append(np.abs(np.fft.fft(raw.astype(float)))[1:21])
                    
                
                
                spectrogram  =  get_binned_spectrogram(raw.astype(float))
                
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
                vad1.set_mode(2)
                
                samples = np.int16(raw)
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
                

                
#                print(feat1,feat2,feat3,feat4,cond5)
                
                if cond1 and cond1_1 and cond2 and cond3 and cond4 and cond5 and cond6:
                    print('VAD',end = '')
                    print(datetime.datetime.now())
#                    plt.imshow(spectrogram)
#                    plt.show()
                
                
                
#                if ratio_egy_consec>0.4:
            
#                print(snd_max.max())
#                if frame_tag.max()>0:
#                    print(ratio_egy_consec)
#                    plt.plot(snd_max)
#                    plt.plot(noise_floor)
#                    plt.plot(frame_tag*snd_max.max() )
#                    plt.show()
                

except KeyboardInterrupt:
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

