N = 2^7;
L = 1;
h = [1/N;1/N;1/N];
epsilon =2/N;


x = linspace(0,1,N);
[XX1,XX2,XX3] = meshgrid(x,x,x);


R = 0.3;

gamma0_1b_f =  @(theta) R*cos(theta)+0.5*(1 + 0.1*cos(4*theta)) ;
gamma0_2b_f = @(theta)  R*sin(theta)+0.5 ;
gamma0_3b_f = @(theta) 0.3*(1 + 0.1*cos(4*theta)) ;


R


K = 200;
LK = K;

theta = linspace(0,2*pi,K+1);
theta = theta(1:K);

gamma0_1 = gamma0_1b_f(theta);
gamma0_2 = gamma0_2b_f(theta);
gamma0_3 = gamma0_3b_f(theta);

k = [0:N/2,-N/2+1:-1];
[K1,K2,K3] = meshgrid(k,k,k);
Delta_F = 4*pi^2*(K1.^2 + K2.^2 + K3.^2);
M = exp(-0.02*epsilon^2*Delta_F);

U = ones(N,N,N);

clf
Ge = compute_geodesic_l_modif(U,M,gamma0_1(2*K/5),gamma0_2(2*K/5),gamma0_3(2*K/5),gamma0_1,gamma0_2,gamma0_3,epsilon);
hold on;
plot3(gamma0_1,gamma0_2,gamma0_3 ,'g',LineWidth=3);


gamma1_1 = gamma0_1(2*K/5)+0.002*cos(theta);
gamma1_2 = gamma0_2(2*K/5)+0.002*sin(theta);
gamma1_3 = gamma0_3(2*K/5)+0*(theta);


plot3(gamma1_1,gamma1_2,gamma1_3,'-r',LineWidth=5);

axis equal


name_fig = ['Test1_gedesique_courbe_point_appro.eps']; 
 print('-depsc', name_fig)