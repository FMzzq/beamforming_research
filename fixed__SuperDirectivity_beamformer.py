# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:38:47 2023
==========================================================================
complete the fixed superdirectivity beamformer
complete in frequency domain
just for study
example
(1) use the uniform_linera_arrays as an example
==========================================================================

@author: zeqingz
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import inv, pinv
#create the microphone array
class uniform_linear_arrays():
    '''
    c : the speed of voice in sky
    n_mic: the number of micphone
    d_mic: the distance between micphone

    '''
    def __init__(self,n_mic,d_mic):
        self.c = 340
        self.n_mic = n_mic
        self.d_mic = d_mic
        self.cut_frq = self.c/2.0/self.d_mic

           
    def steering_vector(self,theta,f):
        '''
        steer_vec [exp(-2j*pi*f*n*d*cos/c) ...    ] n->[0,1...N]
        theta is the angle of desired sig
        f is the frequency of desired sig
        '''
        delay = np.outer(self.d_mic * np.cos(theta) / self.c,np.arange(self.n_mic))   
        steering_vec = np.exp(-2j*np.pi*f*delay)
        return steering_vec
    
class SuperDirectivity():
    '''
    SD beamforming is promising performance in high directivity

    '''
    def __init__(self,array,target_theta,f):
        '''
        array is the microphone arrays 
        the target_theta is the desired angle
        f is the target's frequency
        w is the filter of beamform which we need
        and the out = wH*x
        '''
        self.array = array
        self.target_theta = target_theta
        self.f = f
        self.w = self.create_spatial_filter(self.target_theta,self.f)
        
    def create_spatial_filter(self,target_theta,f):
        steering_vec = self.array.steering_vector(target_theta,f)
        [m_mat, n_mat]  = np.meshgrid(np.arange(self.array.n_mic), np.arange(self.array.n_mic)) 
        mat             = 2 * f * (n_mat - m_mat) * (self.array.d_mic / self.array.c)
        Rnn = np.sinc(mat)
        alpha  = 1e-5
        Rnn = (1 - alpha) * Rnn + alpha * np.identity(self.array.n_mic)
        Rnn_pinv = pinv(Rnn)
        w = np.dot(Rnn_pinv, steering_vec.T) / ( np.conj(steering_vec).dot(Rnn_pinv).dot(steering_vec.T) )
        return w.T.astype(np.complex64)
    
    def spatial_filter_bank(self,D,NFFT,fs):
        '''
        D is the number of directions
        each direction will be has NFFT/2+1 
        '''
        f_range = np.arange(1, NFFT/2+1, 1) / NFFT * fs
        d_range = np.arange(1, 2*D+1, 2)*(np.pi /2/D)
        
        filterbank = np.zeros((f_range.shape[0],d_range.shape[0],self.array.n_mic),dtype = np.complex64)
        for i in range(f_range.shape[0]):
            for j in range(d_range.shape[0]):
                filterbank[i,j,:] = self.create_spatial_filter(d_range[j],f_range[i]).squeeze()
                
        return filterbank
        
             
class beam_analysis():
    '''
    use this class to plot the beampatten of beamformer
    and white-nose -gain and diffuse_noise analysis 
    beamformer : the used beamforming way
    array : the used array type
    '''
    def __init__(self,array,beamformer,beamformer_type):
        self.beamformer = beamformer
        self.array = array
        self.beamformer_type = beamformer_type
    
    def beampatten_plot(self):

        ang_range = np.arange(0, 2*np.pi, np.pi/360)
        streering_vectors = self.array.steering_vector(ang_range,self.beamformer.f)
        w = self.beamformer.w
        response = np.dot(np.conj(w),streering_vectors.T).squeeze()
        response = 20 * np.log10(np.abs(response) + 1e-10)
        ax = plt.subplot(111,projection ='polar')
        title = '{}_M_{}_d_{}_phi_{}_frq_{}'.format(self.beamformer_type,
                                self.array.n_mic, self.array.d_mic, 
                                int(np.degrees(self.beamformer.target_theta)),int(self.beamformer.f))
        ax.set_title(title)
        ax.plot(ang_range,response)
        plt.show()
        
    def white_noise_gain_plot(self):
         '''
         white noise gain is the suppress the white noise of microphone
         G = |wH * x|^2 / wH * Rxx * w  Rxx is the co-variance matrix the white noise 
         accrodiing to the calculate we know the result just dur to the target-angle
         '''
         ang_range = np.arange(0, np.pi, np.pi/180)
         frq_range = np.arange(100, 8000, 100)
         
         array_wng = np.zeros_like(frq_range, dtype=np.float64)
         for i in range(frq_range.shape[0]):
             w = self.beamformer.create_spatial_filter(self.beamformer.target_theta,frq_range[i])
             array_wng[i] = 10 * np.log10(1./(np.abs(np.dot(np.conj(w),w.T).squeeze())+1e-10) + 1e-10)
         ax = plt.subplot(111)
         ax.plot(frq_range ,array_wng)
         plt.show()
         
    def directivity_gain_plot(self):  
         '''
         when the noise is diffuse noise, namely the co-variance matrix is not equal to I
         so accroding to the CSM  sinc(2*pi*f*t) t is the delay between mic so we can calulate 
         the diffuse noise co-matrix
         G = |wH * x|^2 / wH * Rxx * w  Rxx will be  sinc(2*pi*f*t)
         anyway this G due to the target-angle and freqz so we will plot heat plot
         '''
         ang_range = np.arange(0, np.pi, np.pi/180)
         frq_range = np.arange(0, 8000, 100)
         array_divgain = np.zeros((ang_range.shape[0], frq_range.shape[0]), dtype=np.float64)
         [m_mat, n_mat]  = np.meshgrid(np.arange(self.array.n_mic), np.arange(self.array.n_mic)) 
    
         for i in range(ang_range.shape[0]):
             for j in range(frq_range.shape[0]):
                 mat = 2 * frq_range[j] * (n_mat - m_mat) * (self.array.d_mic / self.array.c)
                 Rxx = np.sinc(mat)
                 w = self.beamformer.create_spatial_filter(ang_range[i],frq_range[j])
                 gain = np.dot(np.dot(np.conj(w),Rxx),w.T)
                 array_divgain[i,j] = 10 * np.log10(1./(np.abs(gain.squeeze())) + 1e-10)
         plot = sns.heatmap(array_divgain,cmap = 'gist_rainbow')
         plt.gca().invert_yaxis() 
         plt.show()
           
if __name__ == "__main__":
    ula = uniform_linear_arrays(2,0.15)
    bf = SuperDirectivity(ula,np.pi/2,2000)
    beampatten = beam_analysis(ula,bf,'SuperDirectivity')
    beampatten.beampatten_plot()
    beampatten.white_noise_gain_plot()
    beampatten.directivity_gain_plot()
    filterbank = bf.spatial_filter_bank(3,512,16000)
    np.save('filterbank.npy',filterbank)
    
    
    
    
    
    
    
    