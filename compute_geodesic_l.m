function  Ge = compute_geodesic_l(U,M,gamma1_1,gamma1_2,gamma1_3,gamma0_1,gamma0_2,gamma0_3,epsilon)
K1 = length(gamma1_1);
K = length(gamma0_1);
iplus = [2:K,1];

N = size(U,1);
gamma1_pd = zeros(3,K1);
for i =1:K1 
    gamma1_pd(:,i) = [1 + round(gamma1_1(i)*(N-1)),1 + round(gamma1_2(i)*(N-1)),1 + round(gamma1_3(i)*(N-1))];
end

W =  0.05 + U.^2;
W = real(ifftn(M.*fftn(W)));

for k0 = 1:N
    W(:,:,k0) = W(:,:,k0)';
end

options.nb_iter_max = Inf;
[D,S] = perform_fast_marching(1./W, gamma1_pd,options);


[GD1,GD2,GD3] = gradient(D);
Norm_GD = sqrt(GD1.^2 +GD2.^2 + GD3.^2) + eps;
GD1 = GD1./Norm_GD; GD2 = GD2./Norm_GD; GD3 = GD3./Norm_GD;

sum_Long = 1;
j= 1;

Ge = zeros(size(D));
gamma0_1pd0 = 1 + round(gamma0_1*(N-1));
gamma0_2pd0 = 1 + round(gamma0_2*(N-1));
gamma0_3pd0 = 1 + round(gamma0_3*(N-1));

for i =1:K
Ge(gamma0_2pd0(i),gamma0_1pd0(i),gamma0_3pd0(i)) = 1;
end


%while (max(sqrt((gamma0_1(j,:)-gamma1_1(:)).^2 + ...
 %       (gamma0_2(j,:)-gamma1_2(:)).^2+ (gamma0_3(j,:)-gamma1_3(:)).^2)) >0.1) && j<500
for j=1:100

gamma0_1pd = 1 + round(gamma0_1(j,:)*(N-1));
gamma0_2pd = 1 + round(gamma0_2(j,:)*(N-1));
gamma0_3pd = 1 + round(gamma0_3(j,:)*(N-1));

beta = 1/N;


for i=1:K

grad_D1(i) = GD1(gamma0_1pd(i),gamma0_2pd(i),gamma0_3pd(i));
grad_D2(i) = GD2(gamma0_1pd(i),gamma0_2pd(i),gamma0_3pd(i));
grad_D3(i) = GD3(gamma0_1pd(i),gamma0_2pd(i),gamma0_3pd(i));

gamma0_1(j+1,i) = gamma0_1(j,i) - beta*grad_D2(i) ;
gamma0_2(j+1,i) = gamma0_2(j,i) - beta*grad_D1(i) ;
gamma0_3(j+1,i) = gamma0_3(j,i) - beta*grad_D3(i);

end


%%%%%%%%%%%%%%%%%% redistantiation %%%%%%%%%%%%
Long = sqrt( (gamma0_1(j+1,iplus) - gamma0_1(j+1,:)).^2 + (gamma0_2(j+1,iplus) - gamma0_2(j+1,:)).^2 + (gamma0_3(j+1,iplus) - gamma0_3(j+1,:)).^2) ;
sum_Long = sum(Long(:));

if (min(Long(1:K-1))/max(Long(1:K-1)) < 0.5)  %%%%%%%% on redistancie
j; % pour voir le nombre d'itérarion de l'algo
    
t(1) = 0;   
for i=1:K-1
       t(i+1) = sum(Long(1:i)); 
end


tt = linspace(0,t(K),K);
tt_liste = linspace(0,t(K),floor(16*sum_Long*N/2));

cs = spline(t,gamma0_1(j+1,:));
gamma0_1_liste{j+1}= ppval(cs,tt_liste);
gamma0_1(j+1,:)= ppval(cs,tt);

cs = spline(t,gamma0_2(j+1,:));
gamma0_2_liste{j+1}= ppval(cs,tt_liste);
gamma0_2(j+1,:)= ppval(cs,tt);
cs = spline(t,gamma0_3(j+1,:));
gamma0_3_liste{j+1}= ppval(cs,tt_liste);
gamma0_3(j+1,:)= ppval(cs,tt);


end

j = j+1;

end


for l=1:j-1


gamma0_1pd = 1 + round(gamma0_1_liste{l}*(N-1));
gamma0_2pd = 1 + round(gamma0_2_liste{l}*(N-1));
gamma0_3pd = 1 + round(gamma0_3_liste{l}*(N-1));



for i = 1:length(gamma0_1_liste{l})
    
    Ge(gamma0_2pd(i),gamma0_1pd(i),gamma0_3pd(i)) = 1; 

end

end



%%% On convole avec un noyaux régularisant %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ge = (1/N^2)*real(ifftn(M.*fftn(Ge)));

end