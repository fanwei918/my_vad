#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:39:16 2018

@author: weif
"""


import tempfile
import pandas as pd
from google.cloud import storage
from googleapiclient import discovery
import googleapiclient
import json
import webrtcvad
import struct
import logging

import numpy as np
from scipy.io import wavfile
#import soundfile as sf
import librosa

import keen
from google.cloud import bigquery
import datetime
import time


def histogram_distance_1d(a,b, normalized = False):
    '''
    a,b are two arrays (s.sum() = 1, b.sum() = 1) with equal length = size.
    '''
#   arrays should be normalized：
    if not normalized==True:
        a = a/a.sum()
        b = b/b.sum()
    return np.abs(np.cumsum(a-b)).sum()

    

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
        
        band_cut = band_cut0
        
        band_flatness1[n] =  spectrogram[:band_cut,n].max() /( spectrogram[:band_cut,n].mean()+ 0.00001)  #gmean( spectrogram[:band_cut0,n] ) /  ( spectrogram[:band_cut0,n].mean()+ 0.00001)
        
        max_band = np.argmax(spectrogram[:band_cut,n])
        nbr_max_band = np.arange( max(0,max_band -5) , max_band+5)
        band_flatness[n] =  spectrogram[:band_cut,n].max() /( spectrogram[nbr_max_band,n].mean()+ 0.00001)  #gmean( spectrogram[:band_cut0,n] ) /  ( spectrogram[:band_cut0,n].mean()+ 0.00001)


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
        spectro_similarity[n]= histogram_distance_1d(spectrogram[:,n]+1,spectrogram[:,n+1]+1)
        spectro_similarity_wt[n] = np.minimum( spectrogram[:,n].sum(), spectrogram[:,n+1].sum() )
        
    return spectro_similarity[ np.isnan(spectro_similarity)==False ], spectro_similarity_wt[ np.isnan(spectro_similarity)==False]
        



def vad_1_gcs_trigger_2keen(data, context):
    
    
    print("------------")
    
    bucket_name = data['bucket']
    file_name = data['name']
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(file_name)
    tf = tempfile.NamedTemporaryFile()
    tmp_filename = tf.name
    blob.download_to_filename(tmp_filename)  # save the data to a local temporary file
    
    
    sample_rate = np.int(16000)
    with open(tmp_filename,'rb') as f:
        string = f.read()
    num = len(string)
    fmt = str(num)+'B'
    char_array = np.array( struct.unpack(fmt,string) )
    # convert from char to int
    num = len(char_array)//2
    samples = np.zeros((num,))
    for n in range(num):
        if char_array[2*n+1] >= 128:
            samples[n] = -1 *( (255- char_array[2*n+1])*256 + 256 - char_array[2*n] )
        else:
            samples[n] =  char_array[2*n+1] * 256 +  char_array[2*n]  


    tf.close()
    

    samples = samples.astype(np.int16) / 32768.0
    
    

    if samples.ndim>1 and samples.shape[1]>1:
        samples=samples[:,0]
        
    if sample_rate!=16000:
        samples = (librosa.resample(samples,sample_rate,16000)).astype(float)
        samples = samples/(np.abs(samples).max()+0.00001) *32768
        
    if np.abs(samples).max()<=1:
        samples *= 32768 
    
    spectrogram  =  get_binned_spectrogram(samples.astype(float))
    
    band_flatness, band_flatness1, band_below_ratio1 ,band_below_ratio2, egy_per_frame, egy_focus= vad_condition1(spectrogram, sr = 16000)

    spectro_similarity, spectro_similarity_wt = spectro_simi_measure(spectrogram)
    
    feat1 = np.average(band_flatness,weights = egy_per_frame) 
    feat1_1 = np.average(band_flatness1,weights = egy_per_frame) 
    feat2 = np.average(band_below_ratio1,weights = egy_per_frame)
    feat3 = np.average(band_below_ratio2,weights = egy_per_frame)

    if spectro_similarity_wt.sum()==0:
        feat4 = 1
    else:
        feat4 = np.average(spectro_similarity,weights = spectro_similarity_wt)
        
    feat6 = np.average(egy_focus,weights = egy_per_frame) 
    
    cond1 = feat1>2 and feat1<4
    cond1_1 = feat1_1 > 5 and feat1_1 < 13
    cond2 = feat2 > 0.38
    cond3 = feat3 > 0.18
    cond4 = feat4 > 2.5
    
    cond6 = feat6 > 0.5
    
    
    vad_webrtc = webrtcvad.Vad()
    # set aggressiveness from 0 to 3
    vad_webrtc.set_mode(2)
    
    samples_1 = np.int16(samples)
    raw_samples = struct.pack("%dh" % len(samples_1), *samples_1)
    window_duration = 0.01 # duration in seconds
    samples_per_window = int(window_duration * sample_rate + 0.5)
    bytes_per_sample = 2
    
    segments = []
    speech_per_frame = []

    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
        
        if stop-start< 0.01*sample_rate:
            continue
        
        is_speech = vad_webrtc.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], 
                                  sample_rate = sample_rate)
    
        segments.append(dict(
           start = start,
           stop = stop,
           is_speech = is_speech))
        
        speech_per_frame.append(is_speech)

    detected1 = np.array(speech_per_frame).mean()
    cond5 = detected1>0.2
    
    
    
    vad_detected =0
    if cond1 and cond1_1 and cond2 and cond3 and cond4 and cond5 and cond6:
        vad_detected = 1
    
    
    if vad_detected == 1:
        s = 'Speech Activity'
        logging.critical(s)

    else:
        s = 'Not Speech Activity'
        logging.critical(s)
        
        
    client = bigquery.Client()
    dataset_id = 'VAD_result'  # replace with your dataset ID
    table_id = 'vad_2'  # replace with your table ID
    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)  # API request
    
    t = time.time()
    ts = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    rows_to_insert = [
        (ts, file_name.split('/')[1], file_name.split('/')[2],time.ctime(), s)
    ]
    errors = client.insert_rows(table, rows_to_insert)  # API request
    print(errors)
        
    KEEN_PROJECT_ID = "5ba50a65c9e77c0001c9df86"
    KEEN_WRITE_KEY = "CD7D7BBF9A6E9CC997C3897B4AF4EB0144A0FF14DECD9B1DB7ABCB146D8E7597F49C4C39592E2D27BF0DA3D3D2F31EB3FB9AB7223657272CCC358602123A6A16B82C618B0DC5601BC2AC96013EDF6713C7BA9A8B2354B2C08F1C8CE33296A2BF"
    KEEN_READ_KEY = "0A7B7C68BA6A243960A4315DB65A4E59C27BD61192A71183E10539B1CA061A373B4EF0DEC8577719518D23F5D083D0A0862EA5B7B03FE69701EC874FC6C46B9390505372AB072DA3C3B0C398A630C6E1D44EA368BF40883634430E2DC156CFA1"
    keen.project_id = KEEN_PROJECT_ID
    keen.write_key = KEEN_WRITE_KEY
    keen.read_key = KEEN_READ_KEY
    keen.add_event("vad_classification", {
        "classification_result": s,
        "home_id": file_name.split('/')[1],
        "unit_id": file_name.split('/')[2],
        "timestamp": time.ctime(),
        "cloud_func_no": 1
    })
    
    
    
    
    
    
    