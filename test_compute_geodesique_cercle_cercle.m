N = 2^7;
L = 1;
h = [1/N;1/N;1/N];
epsilon =2/N;


x = linspace(0,1,N);
[XX1,XX2,XX3] = meshgrid(x,x,x);


R = 0.3;

gamma0_1b_f =  @(theta) R*cos(theta)+0.5*(1 + 0*0.1*cos(4*theta)) ;
gamma0_2b_f = @(theta)  R*sin(theta)+0.5 ;
gamma0_3b_f = @(theta) 0.3*(1 + 0*0.1*cos(4*theta)) ;


R = 0.3;

gamma01_1b_f =  @(theta) R*cos(theta).*(1 + 0*0.2*cos(8*theta))+0.5 ;
gamma01_2b_f = @(theta)  R*sin(theta).*(1 + 0*0.2*cos(8*theta))+0.5 ;
gamma01_3b_f = @(theta) 0.7*theta.^0;



K = 200;
LK = K;

theta = linspace(0,2*pi,K+1);
theta = theta(1:K);

gamma0_1 = gamma0_1b_f(theta);
gamma0_2 = gamma0_2b_f(theta);
gamma0_3 = gamma0_3b_f(theta);

%gamma0_1 = [gamma0_1(11:end),gamma0_1(1:10)];
%gamma0_2 = [gamma0_2(11:end),gamma0_2(1:10)];
%gamma0_3 = [gamma0_3(11:end),gamma0_3(1:10)];

gamma1_1 = gamma01_1b_f(theta);
gamma1_2 = gamma01_2b_f(theta);
gamma1_3 = gamma01_3b_f(theta);

%gamma1_1 = [gamma1_1(23:end),gamma1_1(1:22)];
%gamma1_2 = [gamma1_2(23:end),gamma1_2(1:22)];
%gamma1_3 = [gamma1_3(23:end),gamma1_3(1:22)];



k = [0:N/2,-N/2+1:-1];
[K1,K2,K3] = meshgrid(k,k,k);
Delta_F = 4*pi^2*(K1.^2 + K2.^2 + K3.^2);
M = exp(-0.001*epsilon^2*Delta_F);

U = ones(N,N,N);

clf
Ge = compute_geodesic_l_modif(U,M,gamma1_1,gamma1_2,gamma1_3,gamma0_1,gamma0_2,gamma0_3,epsilon);
hold on;
plot3(gamma0_1,gamma0_2,gamma0_3 ,'g',LineWidth=3);
plot3(gamma1_1,gamma1_2,gamma1_3 ,'r',LineWidth=3);

axis equal

name_fig = ['Test1_gedesique_courbe_courbe_appro.eps']; 
 print('-depsc', name_fig)