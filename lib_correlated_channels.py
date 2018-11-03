###############======================##################
### LIB version 1.0   #########
## FOR MODELING INDEPENDENT AND CORRELATED CHANNELS ####
###=================================================### 

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
import sys

def db_to_times_ampl(SNRdB):
    return np.power(10,SNRdB/20.)
def db_to_times_power(SNRdB):
    return np.power(10,SNRdB/10.)

def parsing(name_space, file_name):    
    with open(file_name) as f:
        lines = f.readlines()
    res = np.zeros(len(lines))
    for i in range(len(lines)):
        for j in range(len(name_space)):
            parse = lines[i].find(name_space[j])
            if parse >= 0:
                var_len = len(name_space[j])
                lines[i] = lines[i][var_len:-1]
                lines[i] = lines[i].replace("=", "")
                lines[i] = lines[i].replace(" ", "")
                res[i] = float(lines[i])
    return res


def kron_add(A, B): # Kroneker sum of 2 matrices
    C = np.kron(A, np.eye(len(B))) + \
        np.kron(np.eye(len(A)), B)
    return C
def kron_channels(A, N): # This function generates identical transition matrix for independent identical channels
    # A is a generator matrix for one channel
    # N is number of channels
    if N == 1: # trivial case
        return A
    n = len(A) # number of states for one channel
    ResultSize = np.power(n,N)
    result = zeros((ResultSize,ResultSize))
    Q = zeros((N,n,n))
    for i in range(N):
        Q[i,:,:] = eye(n)  
    for i in range (N):
        Q[i,:,:] = A
        tmp = Q[0,:,:]
        for j in range(N-1): 
#            pdb.set_trace()
            tmp = kron(tmp, Q[j+1,:])
        Q[i,:,:] = eye(n)
        result += tmp        
    return result
#===============================================================================
def ssd(A): # Matrix inverse method Steady-State Distribution
    # A is a transition matrixS
    for i in range(len(A)): # Check the structure of A corresponds to transition matrix 
        if abs(sum(A[i,:])) > abs(A[i,i])/10E5:
            sys.exit("ssd(A) function error: Generator matrix A does not satisfy the requirements")
    N = len(A)
    Atmp = copy(A)
    Atmp = matrix(Atmp)
    Atmp[:,N-1] = ones((N,1))
    return array(Atmp.I[N-1,:])[0]
    
def correlation(SSD, SNR):  # Define the correlation coefficients for all channels                
    # SNR - array of SNR thresholds for every state within one channel 
    n_states = len(SNR) # Number of states per channel
    n_chan = int(math.log(len(SSD),n_states)) # Number of channels
    n_chan_comb = math.factorial(n_chan) # number of channel pairs 12, 23, ...
    n_in_each_state = len(SSD)/n_states # number of times each state appears (needed for EX calculation)
    EX = 0
    DX = 0
 #   pdb.set_trace()   
    for k in range(len(SSD)):   # Mean value
        j = k%n_states
#        print "k= ", k, " j= ",j
        EX += SSD[k]*SNR[j]    
#    pdb.set_trace()   
    for k in range(len(SSD)):   # Dispersion
        j = k%n_states
        DX += SSD[k]*np.power(SNR[j]-EX,2)    
    cov = 0;    # covariation
    
    for k in range(len(SSD)):
        i = k/n_states
        j = k%n_states
#        print i,j,k
#        pdb.set_trace() 
        cov += (SNR[i]-EX)*(SNR[j]-EX)*SSD[k]         
#    pdb.set_trace()
    #===========================================================================
    # for i in range(n_states):
    #     for j in range(n_states):
    #         pdb.set_trace() 
    #         k = i*n_chan+j
    #         cov += (SNR[i]-EX)*(SNR[j]-EX)*SSD[k]     
    #===========================================================================             
    return cov/DX

def state_space(n_channels, n_states):
    n_overall_states = np.power(n_states, n_channels)
    StateSpace = np.chararray(n_overall_states, n_channels)
    for i in range(n_overall_states):
        StateSpace[i]= ad_0(num(i,n_states),n_channels)
    return StateSpace

def channel_statistics(i_channel, str_state_space, SNR, SSD): # Calculates an expectation at i_channel
    # str_state_space - array of strings describing the overall channel states
    n_all_states = size(str_state_space)
    n_states = size(SNR)    # States per channel
    P = zeros(n_states) # Probability vector
    for i_state in range(n_states):
        tmp_p = 0
        for j in range(n_all_states):
            if str_state_space[j][i_channel] == str(i_state):
                P[i_state] +=  SSD[j]
    E = sum(SNR*P)  # Expectation
    Evect = ones(n_states)*E
    VAR = sqrt(abs(sum(np.power((SNR-Evect),2)*P))) # Variance
    return  E, VAR

def mulual_expectation(i_channel, j_channel, str_state_space, SNR, SSD): # Calculates an expectation at i_channel
    # str_state_space - array of strings describing the overall channel states
    n_all_states = size(str_state_space)
    n_states = size(SNR)    # States per channel
    P = zeros(n_states) # Probability vector
    E = 0
#    pdb.set_trace()
    for i_state in range(n_states):
        for j_state in range(n_states):
            p = 0
            for i in range(n_all_states):
                if (str_state_space[i][i_channel] == str(i_state)) and  (str_state_space[i][j_channel] == str(j_state)):
                    p += SSD[i]           
            E += SNR[i_state]*SNR[j_state]*p 
    return  E

def correlation_matrix(A, SNR): # Returns correlation matrix for system with n_channels and n_states per each based on generator matrix A
#    pdb.set_trace()
    n_states = size(SNR)
    n_all_states = len(A)
    n_channels = int(math.log(n_all_states, n_states))
    Correlation_Matrix = np.eye(n_channels)
    SSD = ssd(A)    # Steady-state distribution
    str_state_space = state_space(n_channels, n_states)
    E = zeros(n_channels) # Array of Expectations
    VAR = zeros(n_channels) # Array of Dispersions
    for i_channel in range(n_channels): # Calculating the expectations and variances for all channels
        E[i_channel], VAR[i_channel] = channel_statistics(i_channel, str_state_space, SNR, SSD)
#        pdb.set_trace()
    for i_channel in range(n_channels):
        for j_channel in range(n_channels):
            if i_channel != j_channel:
                Exy = mulual_expectation(i_channel, j_channel, str_state_space, SNR, SSD)
                Correlation_Matrix[i_channel, j_channel] = (Exy-E[i_channel]*E[j_channel])/(VAR[i_channel]*VAR[j_channel])
#    pdb.set_trace()
    test, test1 =linalg.eig(Correlation_Matrix)
    if sum(test>0) < n_channels:
        print "Warning: correlation matrix is not valid"
    return Correlation_Matrix

# With the following function we add the correlation to the matrix A
# A in an initial matrix, obtained by the simple (1+k) multiplication of service rates

def cor_index(state_from, state_to): # Function for add_cor2()
    #--------------------------------------------------------------
    #------- RETURNS INDEX OF CORRELATION -------------------------
    # state_from, state_to - are strings
    if len(state_from) != len(state_to):
        sys.exit("Length of state variables should be equal")
    n_chanels = len(state_from)
    var = zeros(n_chanels)
    tmp = 0
    flag = 0
    i_channel_switch = 0
    for i in range(n_chanels):
        flag = abs(int(state_to[i])-int(state_from[i]))
        tmp+= flag
        if flag == 1:
            i_channel_switch = i
    if tmp != 1 :
        return 0, var      
    for i in range(n_chanels):
        if i == i_channel_switch:
            var[i] = 0
        else: 
            mean_from = (int(state_from[i])+int(state_from[i_channel_switch]))/2.
            var_from = np.power((float(state_from[i])-mean_from),2)
            mean_to = (int(state_to[i])+int(state_to[i_channel_switch]))/2.
            var_to = np.power((float(state_to[i])-mean_to),2)
            var[i] = sign(var_from-var_to)
    return i_channel_switch, var

def num(x, base): # Supplementary function for add_correlation
    mod = x/base
    rem = x%base
    i = 0
    y = 0
    while mod != 0:
        y += rem*np.power(10,i)
        i += 1
        rem  = mod%base
        mod /= base
    y += rem*np.power(10,i)     
    return y      
def ad_0(num, N): # Converts num into string and add zeros at the left if length less than N
    # num - number
    # N - required length of the number     
    str_num = str(num)
    if len(str_num)<N:
        for i in range(N-len(str_num)):
            str_num = '0'+str_num    
    return str_num           
def convert_array_to_matrix(x, n_channels):
    K = len(x)
    k = 0
    corMatrix = eye(n_channels)  
    for i in range(n_channels-1):
        for j in range(i+1,n_channels):
            corMatrix[i,j] = x[k]
            corMatrix[j,i] = x[k]
            k+=1
#    pdb.set_trace()        
    return corMatrix
def convert_matrix_to_array(corMatrix):
    n_channels = len(corMatrix)
    n_pairs = n_channels*(n_channels-1)/2
    vect = zeros(n_pairs)
    k = 0
    for i in range(n_channels-1):
        for j in range(i+1,n_channels):
            vect[k] = corMatrix[i,j]
            k+=1    
    return vect
def valid_cor_matrix_check(correlation_matrix):
    test, test1 =linalg.eig(correlation_matrix)
    if sum(test>0) < len(correlation_matrix):
        print "Correlation matrix is not valid"
        return False
    else:
        return True
        

def add_correlation(A, n_channels, n_states_per_channel, cor): # for different correlations between channels
    # A - matrix of independent channels
    # n_channels - number of channels
    # n_states_per_channel - number of states per channel
    # cor - correlation matrix
    
    if size(cor) == 1: #in case of 2 channels
        # Shaping the correlation matrix
        cor = np.matrix([[1,cor], [cor,1]])    
    else:
        if size(cor.shape) == 1:
            cor = convert_array_to_matrix(cor, n_channels)
        
    Acor = copy(A)
    Acor = np.matrix(Acor)
    n_cor = size(cor) # number of multipliers
    n_pairs = math.factorial(n_channels) # number of channel pairs 12, 23, ...
    for i in range(len(A)):
        st_from = ad_0(str(num(i,n_states_per_channel)),n_channels) # convert i into number of examining channel in string format
        for j in range(len(A)):
            st_to = ad_0(str(num(j,n_states_per_channel)), n_channels) # convert i into number of examining channel in string format
            if st_from!= st_to:
                I, index = cor_index(st_from, st_to)    # I is a number of switched symbol
                for i_channel in range(n_channels):
                    Acor[i, j] *= 1+index[i_channel]*cor[I, i_channel]
    for i in range(len(Acor)): # diagonal elements of Acor
        Acor[i,i] -= sum(Acor[i,:])
    return matrix(Acor)                        
         
        
#===============================================================================
# Modeling Markov channel and save in files back.dat, beta.dat, G.dat
#===============================================================================
def rotate_matrix45(X):
#    pdb.set_trace()
    K = len(X)
    ny = size(X)/K
    Y = zeros((K,K))
    for i in range(K):
        Y[i,i] = X[i,1]
    for i in range(1,K):
        Y[i,i-1] = X[i,0]
    for i in range(K-1):
        Y[i,i+1] = X[i,2]                
    return Y
def Rayleigh_channel(n_states, SNR, ro, fm, Rt, timeSlot):
    # ro - Average SNR
    # eps - parameter required to transform stochastic matrix to a transition matrix
    # fm - % maximum Doppler shift
    # Rt - transmission rate symbol per second
    N = zeros(n_states) # Array of expected numbers of times per sec the received SNR passes downwards across a given SNR levels 
    for i in range(n_states):
        N[i] = sqrt(2*pi*SNR[i]/ro)*fm*exp(-SNR[i]/ro)  
    p ,Rtk = zeros(n_states), zeros(n_states) # vector of transition probabilities
    for i in range(n_states-1):
        p[i] = exp(-SNR[i]/ro) - exp(-SNR[i+1]/ro)
    #p[-1] = 1- sum(p[0:-1])
    p[-1] = exp(-SNR[2]/ro)
    #pdb.set_trace()
    for i in range(n_states):
        #p[i] = exp(-SNR[i-1]/ro) - exp(-SNR[i]/ro)
        Rtk[i] = Rt*p[i]
    #p[-1] = exp(-SNR[-1]/ro) 
    print 'p = {0}'.format(p)
    print 'N = {0}'.format(N)
    # Calculating stochastic matrix
    t = zeros((n_states,n_states))
    for i in range(n_states):
        for j in range(n_states):
            if j-i == 1: # (39)
                t[i,j] = N[i+1]/Rtk[i]
            if j-i == -1: # (40)
                t[i,j] = N[i]/Rtk[i]
    Q = t/timeSlot # Transition matrix
    for i in range(n_states): # Normalization
        t[i,i] = 1-sum(t[i,:])      
        Q[i,i] = -sum(Q[i,:]) 
    g = zeros(n_states)
    for m in range(1,n_states): # States gm
        g[m] = math.log(1+SNR[m],1+SNR[-2])
    return t, Q, g # Transition matrix for one channel


def channel_qualities(SNR, n_channels): # 31/07 backup
    n_states = len(SNR)
    State_Space = state_space(n_channels, n_states)
    n_all_states = len(State_Space)
    G = zeros((n_all_states, n_channels))    
    for i_state in range(n_all_states):
        for i_channel in range(n_channels):
            state = int(State_Space[i_state][i_channel])
            if SNR[state] == 0:
                G[i_state,i_channel] = 0
            else:
                #pdb.set_trace()
                G[i_state,i_channel] = math.log(1+SNR[state],1+SNR[-1])
    return G    
    
def channel_qualities_lin(SNR, n_channels):
    n_states = len(SNR)
    State_Space = state_space(n_channels, n_states)
    n_all_states = len(State_Space)
    G = zeros((n_all_states, n_channels))    
    for i_state in range(n_all_states):
        for i_channel in range(n_channels):
            state = int(State_Space[i_state][i_channel])
            if SNR[state] == 0:
                G[i_state,i_channel] = 0
            else:
                #G[i_state,i_channel] = SNR[state]/SNR[-1]  
                G[i_state,i_channel] = 1
    return G    
  
def overall_channel_condition(G):
    n_states = len(G)
    n_channels = size(G)/len(G)
    beta = zeros(n_states) 
    for i in range(n_states):
        beta[i] = sum(G[i,:])/n_channels
    return beta    
    

def make_generator_matrix_valid(A):
    N = len(A)
    A = abs(A)
    for i in range(N): 
        A[i,i] = - sum(A[i,:]) + A[i,i]
    return A

    
