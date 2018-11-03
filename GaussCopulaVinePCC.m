% 2 channels 3 states each

%close all;
% input parameters
%koef = 0.62*8.1004e-04/16.0689;%/(3.6094e+03);
koef = 60;%1.0082e-05;
rho12 = 0.8; % Correlation
rho23 = 0.7;
rho123 = 0.6;
N = 3;  % Number of users
K = 3;  % Number of states
global P;

% Stochastic matrix for one channel
% t = [[  9.97270385e-01,   2.72961469e-03,   0.00000000e+00],
%        [  6.00446735e-04,   9.98243694e-01,   1.15585884e-03],
%        [  0.00000000e+00,   3.52395166e-04,   9.99647605e-01]];
t = [[  9.99785932e-01,   2.14067869e-04 ,  0.00000000e+00],
                 [  1.42298379e-04 ,  9.99729987e-01 ,  1.27714864e-04],
                 [  0.00000000e+00 ,  1.25331414e-04 ,  9.99874669e-01]];   
% one channel
[v,d] = eig(t);
v = inv(v);
p1 = v(3,:)/ sum(v(3,:))
% Independent  channels
t1 = (t>0);
%T = kron(t,t);T = kron(T,t);
T = kron(t1,t1);T = kron(T,t1);

[V,D] = eig(T);
V = inv(V);
pInd = V(2,:)/ sum(V(2,:));
pInd_marginal = [sum(pInd(1:9)), sum(pInd(10:18)), sum(pInd(19:27))];
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

nStates = K*K*K;
Pst = zeros(nStates);
for iState=1:nStates
    for jState=1:nStates
        a = fV(jState)/fV(iState)*T(jState, iState)/T(iState, jState);
        Pst(iState,jState) = T(iState,jState)*min(1,a);
    end
end

for iState=1:nStates
    Pst(iState,iState) = 1- sum(Pst(iState,:)) + Pst(iState,iState);
end

[V, D] = eig(Pst);
V = inv(V);
pst = V(1,:)/sum(V(1,:));
'Marginal distribution after Generating a Markov chain with correlations'
p_mar = [sum(pst(1:9)), sum(pst(10:18)), sum(pst(19:27))]

A = zeros(nStates);
A = Pst*koef;
A = zeros(nStates);
A = Pst*koef;
for i=1:nStates
A(i,i) = - sum(A(i,:)) + A(i,i);
end
save('A3corRho12_0.8_Rho23_0.7_Rho123_0.6.dat.dat', 'A', '-ASCII');
