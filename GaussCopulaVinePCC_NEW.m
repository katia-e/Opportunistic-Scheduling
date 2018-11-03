% 2 channels 3 states each

%rho12 = 0.8; % Correlation
%rho23 = 0.7;
%rho123 = 0.6;
rho12 = 0.2; % Correlation
rho23 = 0.1;
rho123 = 0.3;
N = 3;  % Number of users
K = 3;  % Number of states
global P;


t = [-509.8584838 509.8584838 0.0000000;
429.9621022 -933.9504191 503.9883168;
0.0000000 177.4689917 -177.4689917]/10;   
% one channel
[v,d] = eig(t);
v = inv(v);
p1 = v(3,:)/ sum(v(3,:))
% Independent  channels
%tp = t.*(ones(3)-eye(3));
%tmp = 0.1;
%for i = 1:3
%    tp(i,:) =  tmp*tp(i,:)/sum(tp(i,:));
%    tp(i,i) = 1-sum(tp(i,:));
%end
%T = kron(tp,tp);T = kron(T,tp);
%[V,D] = eig(T);
%V = inv(V);
%pInd = V(1,:)/ sum(V(1,:));
%pInd_marginal = [sum(pInd(1:9)), sum(pInd(10:18)), sum(pInd(19:27))];
%T = load('A3ind.dat');
P = zeros(1,K); % Marginal cdf
for k=1:K-1  % Marginal cdf
    P(k) = P(k)+p1(k);
    P(k+1) = P(k);
end
P(end) = P(end)+p1(end); % Marginal cdf
copula_type = 'Gaussian';%'Clayton';%   'Gaussian';%'Frank';% 'Clayton';%;'Gumbel'
 F = zeros([K,K,K]); % Joint CDF
 F12 = zeros([K,K,K]); % Joint CDF
 F23 = zeros([K,K,K]); % Joint CDF
% Obtain joint cdf through the matlab function
for i = 1:K
    for j = 1:K
        F12(i,j) = copulacdf(copula_type, [P(i),P(j)], rho12);
        F23(i,j) = copulacdf(copula_type, [P(i),P(j)], rho23);
    end
end
% Obtain joint cdf HAND MADE
% for i = 1:K
%     for j = 1:K
%         for k = 1:K
%             t = cliton(P(i),rho)+cliton(P(j), rho)+cliton(P(k), rho);
%             F(i,j,k) = icliton(t, rho); %copulacdf(copula_type, [P(i),P(j)], rho);
%         end
%     end
% end


f = zeros([K,K,K]); % Joint PDF
% Obtain joint pdf
for y1 = 1:K
    for y2 = 1:K
        for y3 = 1:K
            if (y2 ==1)  denF2 = P(y2);
            else  denF2 = P(y2)-P(y2-1); end;
            for i1=0:1
                for i3=0:1
                    sign = (-1)^(i1+i3);
%                    a = F12(y1-i1, y2) - F12(y1-i1, y2-1);
                    a = copula(y1-i1, y2, rho12) - copula(y1-i1, y2-1, rho12);
                    a = a/denF2;
%                    b = F23(y2, y3-i3) - F23(y2-1, y3-i3);
                    b = copula(y2, y3-i3, rho23) - copula(y2-1, y3-i3, rho23);
                    b = b/denF2;
                    f(y1,y2,y3) = f(y1,y2,y3) + sign*copulacdf(copula_type, [a,b], rho123);
                end
            end
            f(y1,y2,y3) =  f(y1,y2,y3)*denF2;
            
%             if (i==1) && (j==1) && (k==1)  f(i,j,k) = F(i,j,k);
%             elseif (i==1) && (j==1) && (k>1)  f(i,j,k) = F(i,j,k)-F(i,j,k-1);
%             elseif (i==1) && (j>1) && (k==1)  f(i,j,k) = F(i,j,k)-F(i,j-1,k);
%             elseif (i>1) && (j==1) && (k==1)  f(i,j,k) = F(i,j,k)-F(i-1,j,k);
%             elseif (i==1) && (j>1) && (k>1)    f(i,j,k) = F(i,j,k)-F(i,j-1,k)-F(i,j,k-1)+F(i,j-1,k-1);    
%             elseif (i>1) && (j==1) && (k>1)    f(i,j,k) = F(i,j,k)-F(i-1,j,k)-F(i,j,k-1)+F(i-1,j,k-1);
%             elseif (i>1) && (j>1) && (k==1)    f(i,j,k) = F(i,j,k)-F(i-1,j,k)-F(i,j-1,k)+F(i-1,j-1,k);   
%             else
%                 f(i,j,k) = F(i,j,k)-F(i-1,j,k)-F(i,j-1,k)-F(i,j,k-1)+F(i-1,j-1,k)+F(i,j-1,k-1)+F(i-1,j,k-1)-F(i-1,j-1,k-1);
%             end
        end
    end
end
fV = f(:);% reshape(f,[1,K^3])'; % Probability distribution
fV_marginal = [sum(fV(1:9)), sum(fV(10:18)), sum(fV(19:27))]
%[fV(1)+fV(4)+fV(7), fV(2)+fV(5)+fV(8), fV(3)+fV(6)+fV(9)]
% Markov chain Monte-Karlo

nStates = K^3;
a = zeros(nStates);
SNRdb = [0,20,30,40];
SNR = 10.^(SNRdb/10);
%SNR = [31.62,316, 1000];
nu = 1000; Fd = 100;

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
save('AVine.dat', 'a', '-ASCII');
