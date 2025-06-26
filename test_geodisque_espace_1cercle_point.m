%%% Test géodesique dans 
%% On propose dans ce document de déterminer une géodésique directement dans l'espace
%%  de lacets simplifier à cercles de rayon R = x_3 positionné en $ (x_1,0,0)$, espace
%% donc de dimension $2$
%% ON suppose ici que u=1 et donc le poids correspond simplement 
clear all;

figure(1)
R_max = 1;
N = 2^7;
x_R = linspace(-0.5,0.5,N); 
x_1 = linspace(0,1,N);

[X1,X2] = meshgrid(x_1,x_R);
epsilon = 1*10^(-1);
W = (2*pi*abs(X2)+ 0*epsilon./(sqrt(X2.^2 + epsilon)));

gamma1 = [0.5,0.2]; gamma1_pd(1) = 1 + round(gamma1(1)*(N-1));  gamma1_pd(2)= 1 + round((gamma1(2)+0.5)*(N-1));
gamma2 = [0.6,0.4] ; gamma2_pd(1) = 1 + round(gamma2(1)*(N-1));  gamma2_pd(2)= 1 + round((gamma2(2)+0.5)*(N-1));

clf
%colormap('jet')
imagesc(x_1,x_R(N/2+1:N),W(N/2+1:N,:));
hold on;
plot(gamma1(1),gamma1(2),'*g',LineWidth=3);
plot(gamma2(1),gamma2(2),'*r',LineWidth=3);



options.nb_iter_max = Inf;

[D,S] = perform_fast_marching(1./W', gamma1_pd', options);
%Geodesic = round(compute_geodesic(D,gamma2_pd));
%gamma(:,1) = x_1(round(Geodesic(1,:)));
%gamma(:,2) = x_R(round(Geodesic(2,:)));


[GD1,GD2] = gradient(D);
Norm_GD = sqrt(GD1.^2 +GD2.^2 + 0*10^(-16));
GD1 = GD1./Norm_GD; GD2 = GD2./Norm_GD; 

gamma(1,:) = gamma2';

alpha = 0.5/N;
j=1;

%while (max(sqrt((gamma(j,1)-gamma1(1)).^2 + (gamma(j,2)-gamma1(2)).^2+ (gamma(j,3)-gamma1(3)).^2)) >0.05),
for j=1:100,
gamma_pd = 1 + round(gamma(j,:)*(N-1));
gamma_pd(2) = 1 + round((gamma(j,2)+0.5)*(N-1));

grad_D1 = GD1(gamma_pd(1),gamma_pd(2));
grad_D2 = GD2(gamma_pd(1),gamma_pd(2));


gamma(j+1,:) = gamma(j,:) - alpha*[grad_D2,grad_D1];
gamma(j+1,2) = max(gamma(j+1,2),0);

plot(gamma(j+1,1),gamma(j+1,2),'w*')

%j=j+1
end


J = size(gamma,1);

for j=1:J,
plot(gamma(j,1),gamma(j,2),'w*')
end


plot(gamma1(1),gamma1(2),'*g',LineWidth=3);
plot(gamma2(1),gamma2(2),'*r',LineWidth=3);



name_fig = ['Test1_gedesique_poids.eps']; 
 print('-depsc', name_fig)



figure(2)

%%%%%%%%%%%%%%%%%%%%%% on trace les géodésiques %%%%%%%%%%
theta = linspace(0,2*pi);
clf;
plot3(gamma1(1)+gamma1(2)*cos(theta),0.5+gamma1(2)*sin(theta),0+0*theta,'g',LineWidth=3);
hold on;
plot3(gamma2(1)+gamma2(2)*cos(theta),0.5+gamma2(2)*sin(theta),0+0*theta,'r',LineWidth=3);

for j=1:J,
plot3(gamma(j,1)+gamma(j,2)*cos(theta),0.5+gamma(j,2)*sin(theta),0+0*theta,'b');
end

axis auto

name_fig = ['Test1_gedesique_cercle_image_Gamma.eps']; 
 print('-depsc', name_fig)
