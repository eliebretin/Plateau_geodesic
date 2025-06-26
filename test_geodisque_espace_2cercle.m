%%% Test géodesique dans 
%% On propose dans ce document de déterminer une géodésique directement dans l'espace
%%  de lacets simplifier à cercles de rayon R = x_3 positionné en $ (x_1,0,x2)$, espace
%% donc de dimension $3$
%% ON suppose ici que u=1 et donc le poids correspond simplement 
clear all;

R_max = 1;
N = 2^7;
x_R = linspace(-0.5,0.5,N); 
x_1 = linspace(0,1,N);
x_2 = linspace(0,1,N);
[X1,X2,X3] = meshgrid(x_1,x_2,x_R);
epsilon = 1*10^(-2);
W = (2*pi*abs(X3)+ epsilon./(sqrt(X3.^2 + epsilon)));


gamma1 = [0.26,0.5,0.25]; gamma1_pd = 1 + round(gamma1*(N-1));  gamma1_pd(3)= 1 + round((gamma1(3)+0.5)*(N-1));
gamma2 = [0.25,0.8,0.25]; gamma2_pd = 1 + round(gamma2*(N-1));  gamma2_pd(3)= 1 + round((gamma2(3)+0.5)*(N-1));


options.nb_iter_max = Inf;

[D,S] = perform_fast_marching(1./W, gamma1_pd', options);






[GD1,GD2,GD3] = gradient(D);
Norm_GD = sqrt(GD1.^2 +GD2.^2 + GD3.^2);
GD1 = GD1./Norm_GD; GD2 = GD2./Norm_GD; GD3 = GD3./Norm_GD;

gamma(1,:) = gamma2';

alpha = 0.8/N;
j=1;

%while (max(sqrt((gamma(j,1)-gamma1(1)).^2 + (gamma(j,2)-gamma1(2)).^2+ (gamma(j,3)-gamma1(3)).^2)) >0.05),
for j=1:200,
gamma_pd = 1 + round(gamma(j,:)*(N-1));
gamma_pd(3) = 1 + round((gamma(j,3)+0.5)*(N-1));

grad_D1 = GD1(gamma_pd(1),gamma_pd(2),gamma_pd(3));
grad_D2 = GD2(gamma_pd(1),gamma_pd(2),gamma_pd(3));
grad_D3 = GD3(gamma_pd(1),gamma_pd(2),gamma_pd(3));

gamma(j+1,:) = gamma(j,:) - alpha*[grad_D2,grad_D1,grad_D3];
gamma(j+1,3) = max(gamma(j+1,3),0);

%j=j+1
end


J = size(gamma,1);

%%%%%%%%%%%%%%%%%%%%%% on trace les géodésiques %%%%%%%%%%
theta = linspace(0,2*pi);
clf;
plot3(gamma1(1)+gamma1(3)*cos(theta),0.5+gamma1(3)*sin(theta),gamma1(2)+0*theta,'g');
hold on;
plot3(gamma2(1)+gamma2(3)*cos(theta),0.5+gamma2(3)*sin(theta),gamma2(2)+0*theta,'r');

for j=2:J,
plot3(gamma(j,1)+gamma(j,3)*cos(theta),0.5+gamma(j,3)*sin(theta),gamma(j,2)+0*theta,'b');
end

axis equal