# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:19:28 2023\
==================================================================
use the design flter to process the mic wav
anyway the result is on considr

=================================================================
@author: zeqingz
"""
import numpy as np
import soundfile as sf
from scipy import signal
import scipy.io as io
#load window
block_len = 512
block_shift = 160
a_hp = float(0.960998535)
b_pe = float(0.5)
g_input = (float(1.0) + a_hp) / float(2.0)/(float(1.0) + b_pe)
g_output = (float(1.0) + b_pe)
w = io.loadmat(r'E:\New_Research\opt_win_512_512_512_160_3p8.mat')
win_a = np.transpose(w['win_a']).astype(np.float32)[0,:]
win_s = np.transpose(w['win_s']).astype(np.float32)[0,:]
filterbank = np.load('filterbank.npy')
def pre_process_filter(x):
    a_hp = float(0.960998535)
    b_pe = float(0.5)
    g_input = (float(1.0) + a_hp) / float(2.0)/(float(1.0) + b_pe)
    x = x * g_input
    bh = [float(1.0),-b_pe-float(1.0),b_pe]
    ah = [float(1.0),-a_hp]
    y = signal.lfilter(bh,ah,x)   
    return y
def post_process_filter(x):
    b_pe = float(0.5)
    g_output = float(1.5)
    x = x * g_output
    bh = [float(1.0),float(0)]
    ah = [float(1.0),-b_pe]
    y = signal.lfilter(bh,ah,x)   
    return y 
# wav read
audio,fs = sf.read(r'./wav/HF_pub-5db_m0m1.wav',dtype='float32')
org_audio = audio.copy()
out_file = np.zeros(audio.shape[0])
out_file2 = np.zeros(audio.shape[0])
sub = np.zeros(block_shift)
sub2 = np.zeros(block_shift)
# create buffer
#pre-process
 
audio[:,0] = pre_process_filter(audio[:,0])
audio[:,1] = pre_process_filter(audio[:,1])
in_buffer = np.zeros((1,block_len)).astype('float32')
in_buffer2 = np.zeros((1,block_len)).astype('float32')

# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
fn = 0
for idx in range(num_blocks):
    # shift values and write to buffer
    in_buffer[0,:-block_shift] = in_buffer[0,block_shift:]
    in_buffer2[0,:-block_shift] = in_buffer2[0,block_shift:]
    sub = audio[idx*block_shift:(idx*block_shift)+block_shift,0]
    sub2 = audio[idx*block_shift:(idx*block_shift)+block_shift,1]
    in_buffer[0,-block_shift:] = sub
    in_buffer2[0,-block_shift:] = sub2
    in_block_fft = np.fft.rfft(in_buffer*win_a)
    in_block_fft2 = np.fft.rfft(in_buffer2*win_a)
    
    #w*Y
    Y = np.concatenate([in_block_fft,in_block_fft2],axis = 0) #[2,257]
    B = np.zeros_like(in_block_fft,dtype = np.complex128)
    B[0,0] = in_block_fft[0,0]
    for k in range(block_len//2):
        y =  np.conj(filterbank[k,0,:]).dot(Y[:,k+1])
        B[0,k+1] = y

    estimated_block = np.fft.irfft(B)*win_s
    # estimated_block2 = np.fft.irfft(in_block_fft2)*win_s



    # write block to output file
    out_file[0+block_shift*(idx):block_len+block_shift*(idx)] = out_file[0+block_shift*(idx):block_len+block_shift*(idx)] + estimated_block
    # out_file2[0+block_shift*(idx):block_len+block_shift*(idx)] = out_file2[0+block_shift*(idx):block_len+block_shift*(idx)] + estimated_block2
    #de-emphasis
out_file =  post_process_filter(out_file)
# out_file2 =  post_process_filter(out_file2)    
# # write to .wav file
out = np.stack((org_audio[:,1],out_file),axis =-1)
sf.write('./wav/processed.wav', out, fs) 




















