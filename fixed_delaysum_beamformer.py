# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:08:52 2023
==========================================================================
complete the fixed delaysum beamformer
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
    
class delaysum():
    '''
    delay sum beamforming is the most simple 
    and the w is steering_vec / n_mic
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
        self.w = self.create_spatil_filter(self.target_theta,self.f)
        
    def create_spatil_filter(self,target_theta,f):
        steering_vec = self.array.steering_vector(target_theta,f)
        w = steering_vec / self.array.n_mic
        return w

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
         array_wng = np.zeros_like(ang_range, dtype=np.float64)
         for i in range(ang_range.shape[0]):
             w = self.beamformer.create_spatil_filter(ang_range[i],self.beamformer.f)
             array_wng[i] = 10 * np.log10(1./(np.abs(np.dot(np.conj(w),w.T).squeeze())+1e-10) + 1e-10)
         ax = plt.subplot(111)
         ax.plot(ang_range,array_wng)
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
                 w = self.beamformer.create_spatil_filter(ang_range[i],frq_range[j])
                 gain = np.dot(np.dot(np.conj(w),Rxx),w.T)
                 array_divgain[i,j] = 10 * np.log10(1./(np.abs(gain.squeeze())+1e-10) + 1e-10)
         plot = sns.heatmap(array_divgain,cmap = 'gist_rainbow')
         plt.gca().invert_yaxis() 
         plt.show()
           
if __name__ == "__main__":
    ula = uniform_linear_arrays(10,0.04)
    delaysum_bf = delaysum(ula,3*np.pi/4,1500)

    beampatten = beam_analysis(ula,delaysum_bf,'DelaySum')
    beampatten.beampatten_plot()
    beampatten.white_noise_gain_plot()
    beampatten.directivity_gain_plot()

