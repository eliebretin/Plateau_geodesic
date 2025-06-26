function  Ge = compute_geodesic_l_modif(U,M,gamma1_1,gamma1_2,gamma1_3,gamma0_1,gamma0_2,gamma0_3,epsilon)
clf

K1 = length(gamma1_1);
K = length(gamma0_1);
iplus = [2:K,1];

N = size(U,1);
gamma1_pd = zeros(3,K1);
for i =1:K1 
    gamma1_pd(:,i) = [1 + round(gamma1_1(i)*(N-1)),1 + round(gamma1_2(i)*(N-1)),1 + round(gamma1_3(i)*(N-1))];
end


w_min = 0.1;
W = w_min + U.^2;
W = real(ifftn(M.*fftn(W)));

for k0 = 1:N
    W(:,:,k0) = W(:,:,k0)';
end

options.nb_iter_max = Inf;
[D,S] = perform_fast_marching(1./W, gamma1_pd,options);

D = ifftn(M.*fftn(D));

[GD1,GD2,GD3] = gradient(D);
Norm_GD = sqrt(GD1.^2 +GD2.^2 + GD3.^2) + 0.0001;
GD1 = 1*GD1./Norm_GD; GD2 = 1*GD2./Norm_GD; GD3 = 1*GD3./Norm_GD;

GD1 = real(ifftn(M.*fftn(GD1)));
GD2 = real(ifftn(M.*fftn(GD2)));
GD3 = real(ifftn(M.*fftn(GD3)));



sum_Long = 1;
j= 1;

Ge = zeros(size(D));
gamma0_1pd0 = 1 + round(gamma0_1*(N-1));
gamma0_2pd0 = 1 + round(gamma0_2*(N-1));
gamma0_3pd0 = 1 + round(gamma0_3*(N-1));

for i =1:K
Ge(gamma0_2pd0(i),gamma0_1pd0(i),gamma0_3pd0(i)) = 1;
end

gamma0_1_liste{1} =  gamma0_1(:);
gamma0_2_liste{1} =  gamma0_2(:);
gamma0_3_liste{1} =  gamma0_3(:);


D_gamma = 1;



while D_gamma>(2/N)/2 && j<500

gamma0_1pd = 1 + round(gamma0_1_liste{j}*(N-1));
gamma0_2pd = 1 + round(gamma0_2_liste{j}*(N-1));
gamma0_3pd = 1 + round(gamma0_3_liste{j}*(N-1));

K = length(gamma0_1_liste{j});
iplus = [2:K,1];

beta = 0.5/N;
D_gamma = 0;
distance_g_point = 0;
for i=1:length(gamma0_1pd)
    
    
grad_D1(i) = GD1(gamma0_1pd(i),gamma0_2pd(i),gamma0_3pd(i));
grad_D2(i) = GD2(gamma0_1pd(i),gamma0_2pd(i),gamma0_3pd(i));
grad_D3(i) = GD3(gamma0_1pd(i),gamma0_2pd(i),gamma0_3pd(i));

gamma0_1_liste{j+1}(i) = gamma0_1_liste{j}(i) - beta*grad_D2(i) ;
gamma0_2_liste{j+1}(i) = gamma0_2_liste{j}(i) - beta*grad_D1(i) ;
gamma0_3_liste{j+1}(i) = gamma0_3_liste{j}(i) - beta*grad_D3(i) ;

distance_g_point(i) = min(sqrt((gamma0_1_liste{j}(i) - gamma1_1).^2 + (gamma0_2_liste{j}(i) - gamma1_2).^2 + (gamma0_3_liste{j}(i) - gamma1_3).^2));

end


D_gamma = max(distance_g_point);

% 
% 
% 
%%%%%%%%%%%%%%%%%% redistantiation %%%%%%%%%%%%
 Long = sqrt( (gamma0_1_liste{j+1}(iplus) - gamma0_1_liste{j+1}).^2 + (gamma0_2_liste{j+1}(iplus) - gamma0_2_liste{j+1}).^2 ...
     + (gamma0_3_liste{j+1}(iplus) - gamma0_3_liste{j+1}).^2);
 sum_Long = sum(Long(:));
 
 if (min(Long(1:K-1))/max(Long(1:K-1)) < 0.9)  %%%%%%%% on redistancie
 %if mod(j,10)==0, 
 
 t = 0;   
 for i=1:K
        t(i+1) = sum(Long(1:i)); 
 end
 
 
 tt = linspace(0,t(K+1),K+1);
 tt_liste = linspace(0,t(K+1),floor(4*sum_Long*N/2)+4);
 
 
 cs = spline(t,[gamma0_1_liste{j+1},gamma0_1_liste{j+1}(1)]);
 temp =  ppval(cs,tt_liste);
 gamma0_1_liste{j+1}= temp(1:end-1);

 cs = spline(t,[gamma0_2_liste{j+1},gamma0_2_liste{j+1}(1)]);
 temp =  ppval(cs,tt_liste);
 gamma0_2_liste{j+1}= temp(1:end-1);
 
  cs = spline(t,[gamma0_3_liste{j+1},gamma0_3_liste{j+1}(1)]);
 temp =  ppval(cs,tt_liste);
 gamma0_3_liste{j+1}= temp(1:end-1);
 
 

 
 end

  plot3(gamma0_1_liste{j},gamma0_2_liste{j},gamma0_3_liste{j},'b');
  hold on;


j = j+1;

end

j
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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