%% Single channel parameters
nu = 700;                          % Mean channel SNR
Fd = 100;                           % Doppler frequency
%% Multy-channel environment
K = 3;                                  % Number of customers
tetta = 1;                             % correlation parameter of Clyton
%SNRdb = [0,23.05,27.435,40];     % SNR thresholds, dB  
SNRdb = [0,23,27,40];     % SNR thresholds, dB  
%SNRdb = [0,20,27,40];
%% Single channel precalculation
SNR = 10.^(SNRdb/10);   % SNR thresholds, Xtimes
SNR(1) = 0.00001;
pdf = zeros(1,K);               % Probability dencity function for a channel
for i=1:K
    pdf(i) = exp(-SNR(i)/nu)-exp(-SNR(i+1)/nu);
end
pdf
cdf = 1-exp(-SNR(2:end)./nu);      % Cumulative distribution function for one channel
%% Multichannel Clyton model
F = zeros([K,K,K]);             % Cumulative distribution function for multy-channel model
for i=1:K
    for j=1:K
        for k=1:K
            tmp = cdf(i)^(-tetta)+cdf(j)^(-tetta)+cdf(k)^(-tetta)-2;
            F(i,j,k) = max(tmp,0)^(-1/tetta);
        end
    end
end
f = zeros([K,K,K]);             % Probability dencity function for multy-channel model
for i=1:K
    for j=1:K
        for k=1:K
            f(i,j,k) = F(i,j,k);
            if i>1  f(i,j,k) = f(i,j,k)-F(i-1,j,k); end
            if j>1  f(i,j,k) = f(i,j,k)-F(i,j-1,k); end     
            if k>1  f(i,j,k) = f(i,j,k)-F(i,j,k-1); end
            if i>1 && j>1  f(i,j,k) = f(i,j,k)+F(i-1,j-1,k); end
            if i>1 && k>1  f(i,j,k) = f(i,j,k)+F(i-1,j,k-1); end
            if j>1 && k>1  f(i,j,k) = f(i,j,k)+F(i,j-1,k-1); end
            if i>1 && k>1 &&j>1 f(i,j,k) = f(i,j,k)-F(i-1,j-1,k-1); end
        end
    end
end
fV = f(:);
fV_marginal = [sum(fV(1:9)), sum(fV(10:18)), sum(fV(19:27))];
nStates = K^3;
a = zeros(nStates);

for iFROM=1:K
     for iTO=1:K
        for jFROM=1:K
            for jTO=1:K
                for kFROM=1:K
                    for kTO=1:K        
                        snr = 0;
                        chanStateFROM = (iFROM-1)*K^2+(jFROM-1)*K+kFROM;
                        chanStateTO = (iTO-1)*K^2+(jTO-1)*K+kTO;
                        if (iTO-iFROM) == 1&& (jTO-jFROM)==0 && (kTO-kFROM) == 0
                            snr = SNR(iTO);
                        end
                        if (iTO-iFROM) == -1&& (jTO-jFROM)==0 && (kTO-kFROM) == 0
                            snr = SNR(iFROM);
                        end                        
                        if (iTO-iFROM) == 0&& (jTO-jFROM)==1 && (kTO-kFROM) == 0
                            snr = SNR(jTO);
                        end
                        if (iTO-iFROM) == 0&& (jTO-jFROM)==-1 && (kTO-kFROM) == 0
                            snr = SNR(jFROM);
                        end  
                        if (iTO-iFROM) == 0&& (jTO-jFROM)==0 && (kTO-kFROM) == 1
                            snr = SNR(kTO);
                        end
                        if (iTO-iFROM) == 0&& (jTO-jFROM)==0 && (kTO-kFROM) == -1
                            snr = SNR(kFROM);
                        end                           
                        a(chanStateFROM,chanStateTO) = sqrt(2*pi*snr/nu)*Fd*exp(-snr/nu) /fV(chanStateFROM);
                    end
                end
             end
        end
    end
end

for i = 1:nStates
    a(i,i) = a(i,i) - sum(a(i,:));
end
[V, D] = eig(a);
aTms = [a(:,1:end-1), ones(nStates, 1)];
aTms = inv(aTms);
P = aTms(end,:);
p_mar = [sum(P(1:9)), sum(P(10:18)), sum(P(19:27))]
a = a/10;
save('ACyton.dat', 'a', '-ASCII');


