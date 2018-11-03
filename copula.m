function x = copula(i,j, rho)
    global P;  
    if (i<1)|(j<1)
        x = 0;
    else
        x = copulacdf('Gaussian', [P(i),P(j)], rho);
    end
end

