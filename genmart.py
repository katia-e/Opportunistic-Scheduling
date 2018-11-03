### script version 1.0   #########
###########################################################
#######
#######   THIS SCRIPT GENERATE TRANSITION MATRICES
#######  FOR BOTH INDEPENDENT AND CORRELATED CHANNELS
#######
###########################################################
#######     RUNNING INSTRUCTIONS: python genmatr.py k eps
#######     with k=2(def) number of channels and eps=1(def) and additional coef
import pdb # pdb.set_trace()
from numpy import *
import numpy as np
from numpy.random import *
import itertools
import datetime
import time
import math
from numpy.linalg import inv
#from pylab import * # plot
from lib_correlated_channels import * 
from random import randint
#from scipy.special import jv
#import msvcrt as m

def get_correlation_matrix(file_name_correlation_matrix_input):
    file_with_matrix = open ( file_name_correlation_matrix_input , 'r')
    Correlation_Matrix = np.loadtxt(file_with_matrix)
    return Correlation_Matrix
correlations_are_there = False
DIR_OUTPUT = 'MATRICES/'
def main():       
############################
##### INPUT ################
############################
    if len(sys.argv) > 1:
        n_channels = int(sys.argv[1])
    else:
        n_channels = 2
    if len(sys.argv) > 2:
        eps = int(sys.argv[2])
    else:
        eps = 1            
    #pdb.set_trace()
    DIR_INPUT_PARAMETERS = ""
    #===============================================================
    fm = 100    # Max Doppler spread
    Rt = 1E7#4E6 # Transmission rate symbol per second
    #SNRdB = np.array([30, 40, 100])
    SNRdB = np.array([15, 25, 30])
    #SNRdB = np.array([20, 21, 22])   
    #SNRdB = np.array([20, 30, 35]) 
    rodB = 30   # Expected SNR value in dB    
    #===============================================================
    ##### SOURCE FOLDERS ########
    #file_name_A = DIR_INPUT_PARAMETERS + 'back.dat'
    file_name_beta = DIR_INPUT_PARAMETERS + 'beta.dat'
    file_name_G = DIR_INPUT_PARAMETERS + 'G.dat'
    file_name_SNR = DIR_INPUT_PARAMETERS + 'snr.dat'
    file_name_channel_info = DIR_INPUT_PARAMETERS + 'channel_info.dat'
    file_name_correlation_matrix_input = DIR_INPUT_PARAMETERS + 'Correlation_Matrix.dat'
    file_name_correlation_matrix_output = DIR_INPUT_PARAMETERS + 'Correlation_Matrix_Out.dat'
    file_name_eigenvalues =  DIR_INPUT_PARAMETERS + 'eigenvalues.dat'
    file_name_eigenvectors =  DIR_INPUT_PARAMETERS + 'eigenvectors.dat'

################################
########   CODE   ##############
################################    

#    WHEN DEBAGING DEFINE THE NUMBER OF CHANNELS
#    n_channels = 2    
    if correlations_are_there:
        init_cor_matrix = get_correlation_matrix(file_name_correlation_matrix_input)
        n_channels = len(init_cor_matrix)   
        if valid_cor_matrix_check(init_cor_matrix) == False: ## <<<< Check if correlation matrix is valid
            sys.exit("Termination: Correlation matrix is not positive semidefinite)")
            wait()
        init_cor_vector = convert_matrix_to_array(init_cor_matrix)
#    PARAMETERS FOR SINGE CHANNEL MODEL     
    SNR = db_to_times_power(SNRdB)	    
    ro = db_to_times_power(rodB) 
    np.savetxt('SNR1.dat', SNR, fmt='%-7.2f')
    n_states = size(SNR)   
    if  (SNRdB[0] == 0):
        SNR[0] = 0
    G = channel_qualities(SNR, n_channels)
    np.savetxt(file_name_G, G, fmt='%-7.2f')
    beta = overall_channel_condition(G)
    #pdb.set_trace()
    timeSlot = 1
    t, a, g = Rayleigh_channel(n_states, SNR, ro, fm, Rt, timeSlot)
    a = Rt*a # scaling a
    print 'SNR = {0}, rho = {1}={2}bB'.format(SNR, ro, rodB)
    print 'Stochastic matrix\n {0},\n Generator matrix\n {1}'.format(t,a)
    #print 'channel qualities\n, ', g
    
    np.savetxt('TransMatr.dat', t, fmt='%-7.7f')
    A = kron_channels(a, n_channels)
    ################################

    result_A = A
    file_name_A = 'A.dat'
    print "Transition rate matrix without correlations is ready\n"
    print "DONE"     
    #  Print result A to the file
    np.savetxt('DTtransRateMatrix.dat', t, fmt='%-7.7f') # Save single channel stochastic matrix
    np.savetxt('CTsingleChanGenMart.dat', a, fmt='%-7.7f') # Save resulting generator matrix
    np.savetxt(file_name_A, result_A, fmt='%-10.10f') # Save resulting generator matrix	
    #np.savetxt('G.dat', g, fmt='%-7.7f') # Save resulting generator matrix
    #  Print eigenvectors and eigenvalues to the file
    #eigen = linalg.eig(result_A)
    #np.savetxt(file_name_eigenvalues, eigen[0], fmt='%-7.4f') # Save matrix with correlations
    #np.savetxt(file_name_eigenvectors, eigen[1], fmt='%-7.4f') # Save matrix with correlations
    #print result_A
##########################
main()



###################

#    init_cor_matrix = np.matrix([[1, 0.9 ],
#                          [0.9, 1]])    
#            
    #===========================================================================
    #===========================================================================
    # init_cor_matrix = np.matrix([[1, 0, 0.9],
    #                             [0, 1, 0.0],
    #                             [0.9, 0.0, 1]])
    #===========================================================================
    
    #===========================================================================
    #===========================================================================
    #    init_cor_matrix = np.matrix([[1, 0.4, 0.1, 0.0],
    #                                 [0.4, 1, 0.1, 0.0],
    #                                 [0.1, 0.1, 1, 0.0],
    #                                 [0.0, 0.0, 0.0, 1]])
    
    
    #  Save to files    
    #===========================================================================
    # channel_info = ["System with " + str(n_channels) + " channels with " + str(n_states)+ " states per each",
    #                 channel_model+" channel model",
    #                 "SNR levels: "+ str(SNRdB) + "dB (power)",
    #                 "Max Doppler spread " + str(fm),
    #                 "Average SNR "+ str(ro)+ "dB (power)"]
    # info_file = open("Data/channel_info.txt", "w")
    # for i in range(len(channel_info)): info_file.write(channel_info[i]+"\n")
    # info_file.close()
    #===========================================================================
    #np.savetxt('Data/back.dat', A, fmt='%-7.2f')    
    
    #    SSD = ssd(A)
#    SSDcor = ssd(Acor)
#    average_channel_quality = sum(beta*SSD)
#    average_channel_qualitycor = sum(beta*SSDcor)
     
#    pdb.set_trace()  