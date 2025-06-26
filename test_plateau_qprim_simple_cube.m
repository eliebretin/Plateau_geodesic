close all;
clear all;
clf;


N = 2^6;
L = 1;
h = [1/N;1/N;1/N];
epsilon =2/N;


  vv = VideoWriter(['test_plateau_cube.avi'],'Motion JPEG AVI');
   vv.Quality = 90;
   open(vv);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Geodesic %%%%%%%%%%%%%ù
x = linspace(0,1,N);
[XX1,XX2,XX3] = meshgrid(x,x,x);

gamma1_1f = @(theta)  0.1 + 0.8*[0.01+ 0.98*theta 0.99*ones(size(theta)) 0.99-0.98*theta 0.01*ones(size(theta))] ;
gamma1_2f = @(theta)  0.1 + 0.8*zeros(1,4*length(theta));
gamma1_3f = @(theta)  0.1 + 0.8*[0.01*ones(size(theta)) 0.01 + 0.98*theta 0.99*ones(size(theta)) 0.99-0.98*theta ] ;

gamma2_1f = @(theta)  0.1 + 0.8*[0.99+0.01*theta ones(size(theta)) 1-0.01*theta 0.99*ones(size(theta))];
gamma2_2f = @(theta)  0.1 + 0.8*[theta ones(size(theta)) (1-theta) zeros(size(theta))] ;
gamma2_3f = @(theta)  0.1 + 0.8*[0.01 -0.01*theta theta 1-0.01*theta 0.01+ 0.98*(1-theta) ] ;

gamma3_1f = @(theta)  0.1 + 0.8*[theta ones(size(theta)) 1-theta zeros(size(theta))] ;
gamma3_2f = @(theta)  0.1 + 0.8*ones(1,4*length(theta));
gamma3_3f = @(theta)  0.1 + 0.8*[zeros(size(theta)) theta ones(size(theta)) 1-theta ] ;

gamma4_1f = @(theta)  0.1 + 0.8*[0.01*(1-theta) zeros(size(theta)) 0.01*theta 0.01*ones(size(theta))];
gamma4_2f = @(theta)  0.1 + 0.8*[theta ones(size(theta)) (1-theta) zeros(size(theta))] ;
gamma4_3f = @(theta)  0.1 + 0.8*[0.01 -0.01*theta theta 1-0.01*theta 0.01+ 0.98*(1-theta) ]  ;

%gamma5_1f = @(theta)  0.1 + 0.8*[theta ones(size(theta)) 1-theta zeros(size(theta))] ;
%gamma5_2f = @(theta)  0.15 + 0.75*[zeros(size(theta)) theta ones(size(theta)) 1-theta ] ;
%gamma5_3f = @(theta)  0.1 + 0.8*zeros(1,4*length(theta));
% 
gamma6_1f = @(theta)  0.1 + 0.8*[theta ones(size(theta)) 1-theta zeros(size(theta))] ;
gamma6_2f = @(theta)  0.1 + 0.8*[zeros(size(theta)) theta ones(size(theta)) 1-theta ] ;
gamma6_3f = @(theta)  0.1 + 0.8*ones(1,4*length(theta));


%%%%%%%%%%%%%%%%%%%%%%% Pour l'affichage %%%%%%%%%%%%%%%%%%%%%%%%%%
Kb = 500; 
theta = linspace(0,1,Kb+1);
theta = theta(1:Kb);

gamma2_1b = gamma2_1f(theta); gamma2_2b = gamma2_2f(theta); gamma2_3b = gamma2_3f(theta);
gamma1_1b = gamma1_1f(theta); gamma1_2b = gamma1_2f(theta); gamma1_3b = gamma1_3f(theta);
gamma3_1b = gamma3_1f(theta); gamma3_2b = gamma3_2f(theta); gamma3_3b = gamma3_3f(theta);
gamma4_1b = gamma4_1f(theta); gamma4_2b = gamma4_2f(theta); gamma4_3b = gamma4_3f(theta);
%gamma5_1b = gamma5_1f(theta); gamma5_2b = gamma5_2f(theta); gamma5_3b = gamma5_3f(theta);
gamma6_1b = gamma6_1f(theta); gamma6_2b = gamma6_2f(theta); gamma6_3b = gamma6_3f(theta);

% gamma7_1b = gamma1_1b(floor(Kb/2)); gamma7_2b = gamma1_2b(floor(Kb/2)); gamma7_3b = gamma1_3b(floor(Kb/2));
% gamma8_1b = gamma2_1b(floor(Kb/2)); gamma8_2b = gamma2_2b(floor(Kb/2)); gamma8_3b = gamma2_3b(floor(Kb/2));
% gamma9_1b = gamma3_1b(floor(Kb/2)); gamma9_2b = gamma3_2b(floor(Kb/2)); gamma9_3b = gamma3_3b(floor(Kb/2));
% gamma10_1b = gamma4_1b(floor(Kb/2)); gamma10_2b = gamma4_2b(floor(Kb/2)); gamma10_3b = gamma4_3b(floor(Kb/2));
% %gamma11_1b = gamma5_1b(floor(Kb/2)); gamma11_2b = gamma5_2b(floor(Kb/2)); gamma11_3b = gamma5_3b(floor(Kb/2));
% gamma12_1b = gamma6_1b(floor(Kb/2)); gamma12_2b = gamma6_2b(floor(Kb/2)); gamma12_3b = gamma6_3b(floor(Kb/2));
% gamma13_1b = gamma6_1b(Kb+floor(Kb/2)); gamma13_2b = gamma6_2b(Kb+floor(Kb/2)); gamma13_3b = gamma6_3b(Kb+floor(Kb/2));
% gamma14_1b = gamma6_1b(2*Kb+floor(Kb/2)); gamma14_2b = gamma6_2b(2*Kb+floor(Kb/2)); gamma14_3b = gamma6_3b(2*Kb+floor(Kb/2));
% gamma15_1b = gamma6_1b(3*Kb+floor(Kb/2)); gamma15_2b = gamma6_2b(3*Kb+floor(Kb/2)); gamma15_3b = gamma6_3b(3*Kb+floor(Kb/2));

gamma7_1b = gamma1_1b(1); gamma7_2b = gamma1_2b(1); gamma7_3b = gamma1_3b(1);
gamma8_1b = gamma2_1b(1); gamma8_2b = gamma2_2b(1); gamma8_3b = gamma2_3b(1);
gamma9_1b = gamma3_1b(1); gamma9_2b = gamma3_2b(1); gamma9_3b = gamma3_3b(1);
gamma10_1b = gamma4_1b(1); gamma10_2b = gamma4_2b(1); gamma10_3b = gamma4_3b(1);
gamma12_1b = gamma6_1b(1); gamma12_2b = gamma6_2b(1); gamma12_3b = gamma6_3b(1);
gamma13_1b = gamma6_1b(Kb); gamma13_2b = gamma6_2b(Kb); gamma13_3b = gamma6_3b(Kb);
gamma14_1b = gamma6_1b(2*Kb); gamma14_2b = gamma6_2b(2*Kb); gamma14_3b = gamma6_3b(2*Kb);
gamma15_1b = gamma6_1b(3*Kb); gamma15_2b = gamma6_2b(3*Kb); gamma15_3b = gamma6_3b(3*Kb);

%%%%%%%%%%%%%%%%%%%%%%% Pour le calcul des géodésiques

K = 100; LK = K;
theta = linspace(0,1,K+1); theta = theta(1:K);

gamma2_1 = gamma2_1f(theta); gamma2_2 = gamma2_2f(theta); gamma2_3 = gamma2_3f(theta);
gamma1_1 = gamma1_1f(theta); gamma1_2 = gamma1_2f(theta); gamma1_3 = gamma1_3f(theta);
gamma3_1 = gamma3_1f(theta); gamma3_2 = gamma3_2f(theta); gamma3_3 = gamma3_3f(theta);
gamma4_1 = gamma4_1f(theta); gamma4_2 = gamma4_2f(theta); gamma4_3 = gamma4_3f(theta);
% gamma5_1 = gamma5_1f(theta); gamma5_2 = gamma5_2f(theta); gamma5_3 = gamma5_3f(theta);
 gamma6_1 = gamma6_1f(theta); gamma6_2 = gamma6_2f(theta); gamma6_3 = gamma6_3f(theta);

% gamma7_1 = gamma1_1(floor(K/2)); gamma7_2 = gamma1_2(floor(K/2)); gamma7_3 = gamma1_3(floor(K/2));
% gamma8_1 = gamma2_1(floor(K/2)); gamma8_2 = gamma2_2(floor(K/2)); gamma8_3 = gamma2_3(floor(K/2));
% gamma9_1 = gamma3_1(floor(K/2)); gamma9_2 = gamma3_2(floor(K/2)); gamma9_3 = gamma3_3(floor(K/2));
% gamma10_1 = gamma4_1(floor(K/2)); gamma10_2 = gamma4_2(floor(K/2)); gamma10_3 = gamma4_3(floor(K/2));
% %gamma11_1 = gamma5_1(floor(K/2)); gamma11_2 = gamma5_2(floor(K/2)); gamma11_3 = gamma5_3(floor(K/2));
% gamma12_1 = gamma6_1(floor(K/2)); gamma12_2 = gamma6_2(floor(K/2)); gamma12_3 = gamma6_3(floor(K/2));
% gamma13_1 = gamma6_1(K+floor(K/2)); gamma13_2 = gamma6_2(K+floor(K/2)); gamma13_3 = gamma6_3(K+floor(K/2));
% gamma14_1 = gamma6_1(2*K+floor(K/2)); gamma14_2 = gamma6_2(2*K+floor(K/2)); gamma14_3 = gamma6_3(2*K+floor(K/2));
% gamma15_1 = gamma6_1(3*K+floor(K/2)); gamma15_2 = gamma6_2(3*K+floor(K/2)); gamma15_3 = gamma6_3(3*K+floor(K/2));

gamma7_1 = gamma1_1(1); gamma7_2 = gamma1_2(1); gamma7_3 = gamma1_3(1);
gamma8_1 = gamma2_1(1); gamma8_2 = gamma2_2(1); gamma8_3 = gamma2_3(1);
gamma9_1 = gamma3_1(1); gamma9_2 = gamma3_2(1); gamma9_3 = gamma3_3(1);
gamma10_1 = gamma4_1(1); gamma10_2 = gamma4_2(1); gamma10_3 = gamma4_3(1);
gamma12_1 = gamma6_1(1); gamma12_2 = gamma6_2(1); gamma12_3 = gamma6_3(1);
gamma13_1 = gamma6_1(K); gamma13_2 = gamma6_2(K); gamma13_3 = gamma6_3(K);
gamma14_1 = gamma6_1(2*K); gamma14_2 = gamma6_2(2*K); gamma14_3 = gamma6_3(2*K);
gamma15_1 = gamma6_1(3*K); gamma15_2 = gamma6_2(3*K); gamma15_3 = gamma6_3(3*K);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
 Domaine1_tube =  zeros(N,N,N);

 X1_tube = zeros(N,N,N); 
 length_X1_tube = 0.3;
 for i=1:length(gamma1_2b)
 X1_tube = max(X1_tube,exp(-pi*length_X1_tube*((( gamma1_2b(1,i)-XX2).^2 + ( gamma1_1b(1,i)-XX1).^2 + ( gamma1_3b(1,i)-XX3).^2))/(epsilon^2)));
 end  


 X2_tube = zeros(N,N,N);
 length_X2_tube = 0.3;
 for i=1:length(gamma2_2b)
 X2_tube = max(X2_tube,exp(-pi*length_X2_tube*((( gamma2_2b(1,i)-XX2).^2 + ( gamma2_1b(1,i)-XX1).^2 + ( gamma2_3b(1,i)-XX3).^2))/(epsilon^2)));
 end  


  X3_tube = zeros(N,N,N);
 length_X3_tube = 0.3;
 for i=1:length(gamma3_2b)
 X3_tube = max(X3_tube,exp(-pi*length_X3_tube*((( gamma3_2b(1,i)-XX2).^2 + ( gamma3_1b(1,i)-XX1).^2 + ( gamma3_3b(1,i)-XX3).^2))/(epsilon^2)));
 end  


 X4_tube = zeros(N,N,N);
 length_X4_tube = 0.3;
 for i=1:length(gamma4_2b)
 X4_tube = max(X4_tube,exp(-pi*length_X4_tube*((( gamma4_2b(1,i)-XX2).^2 + ( gamma4_1b(1,i)-XX1).^2 + ( gamma4_3b(1,i)-XX3).^2))/(epsilon^2)));
 end  

%  X5_tube = zeros(N,N,N);
%  length_X5_tube = 0.3;
%  for i=1:length(gamma5_2b)
%  X5_tube = max(X5_tube,exp(-pi*length_X5_tube*((( gamma5_2b(1,i)-XX2).^2 + ( gamma5_1b(1,i)-XX1).^2 + ( gamma5_3b(1,i)-XX3).^2))/(epsilon^2)));
%  end  
% 
%  X6_tube = zeros(N,N,N);
%  length_X6_tube = 0.3;
%  for i=1:length(gamma6_2b)
%  X6_tube = max(X6_tube,exp(-pi*length_X6_tube*((( gamma6_2b(1,i)-XX2).^2 + ( gamma6_1b(1,i)-XX1).^2 + ( gamma6_3b(1,i)-XX3).^2))/(epsilon^2)));
%  end  

 X7_tube = zeros(N,N,N);
 length_X7_tube = 0.2;
 for i=1:length(gamma7_2b)
 X7_tube = max(X7_tube,exp(-pi*length_X7_tube*((( gamma7_2b(1,i)-XX2).^2 + ( gamma7_1b(1,i)-XX1).^2 + ( gamma7_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

 X8_tube = zeros(N,N,N);
 length_X8_tube = 0.2;
 for i=1:length(gamma8_2b)
 X8_tube = max(X8_tube,exp(-pi*length_X8_tube*((( gamma8_2b(1,i)-XX2).^2 + ( gamma8_1b(1,i)-XX1).^2 + ( gamma8_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

 X9_tube = zeros(N,N,N);
 length_X9_tube = 0.2;
 for i=1:length(gamma9_2b)
 X9_tube = max(X9_tube,exp(-pi*length_X9_tube*((( gamma9_2b(1,i)-XX2).^2 + ( gamma9_1b(1,i)-XX1).^2 + ( gamma9_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

 X10_tube = zeros(N,N,N);
 length_X10_tube = 0.2;
 for i=1:length(gamma10_2b)
 X10_tube = max(X10_tube,exp(-pi*length_X10_tube*((( gamma10_2b(1,i)-XX2).^2 + ( gamma10_1b(1,i)-XX1).^2 + ( gamma10_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

%  X11_tube = zeros(N,N,N);
%  length_X11_tube = 0.2;
%  for i=1:length(gamma11_2b)
%  X11_tube = max(X11_tube,exp(-pi*length_X11_tube*((( gamma11_2b(1,i)-XX2).^2 + ( gamma11_1b(1,i)-XX1).^2 + ( gamma11_3b(1,i)-XX3).^2))/(epsilon^2)));
%  end

 X12_tube = zeros(N,N,N);
 length_X12_tube = 0.2;
 for i=1:length(gamma12_2b)
 X12_tube = max(X12_tube,exp(-pi*length_X12_tube*((( gamma12_2b(1,i)-XX2).^2 + ( gamma12_1b(1,i)-XX1).^2 + ( gamma12_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

 X13_tube = zeros(N,N,N);
 length_X13_tube = 0.2;
 for i=1:length(gamma13_2b)
 X13_tube = max(X13_tube,exp(-pi*length_X13_tube*((( gamma13_2b(1,i)-XX2).^2 + ( gamma13_1b(1,i)-XX1).^2 + ( gamma13_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

 X14_tube = zeros(N,N,N);
 length_X14_tube = 0.2;
 for i=1:length(gamma14_2b)
 X14_tube = max(X14_tube,exp(-pi*length_X14_tube*((( gamma14_2b(1,i)-XX2).^2 + ( gamma14_1b(1,i)-XX1).^2 + ( gamma14_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

 X15_tube = zeros(N,N,N);
 length_X15_tube = 0.2;
 for i=1:length(gamma15_2b)
 X15_tube = max(X15_tube,exp(-pi*length_X15_tube*((( gamma15_2b(1,i)-XX2).^2 + ( gamma15_1b(1,i)-XX1).^2 + ( gamma15_3b(1,i)-XX3).^2))/(epsilon^2)));
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k = [0:N/2,-N/2+1:-1]; [K1,K2,K3] = meshgrid(k,k,k); Delta = -4*pi^2*(K1.^2 + K2.^2 + K3.^2); 
Mreg = exp(+0.01*epsilon^2*Delta);
U = ones(N,N,N);

Ge1 = compute_geodesic_l_modif(U,Mreg,gamma10_1,gamma10_2,gamma10_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
Ge2 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
Ge3 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
Ge4 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% Ge5 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% Ge6 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma6_1,gamma6_2,gamma6_3,epsilon);


Ge =max(max(max(Ge1,Ge2),Ge3),Ge4);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u = 0.1*0.25*Ge/max(Ge(:));


F_prim = @(s) (1 - 6*s).*s;
F_seconde = @(s) (1 - 12*s);
dt =10*epsilon^2;
alpha =0/epsilon^2;
beta = 2/epsilon^0;
sigma =1;
M = 1./(1 + dt*( 1*sigma*epsilon^2*Delta.^2  - Delta  +   alpha - beta*Delta));
j_sauvegarde  = 1;

Mreg = exp(+0.1*epsilon^2*Delta);


T = 1 ;
N_T = 10;
T_vec = linspace(0,T*(1.01),N_T);
j_vec = 1;




for i=1:T/dt
gamma_epsilon =150/epsilon^3;

    
    dphi = Ge.*(sqrt(1-4*u));
    Delta_gamma = dphi; 
    omega = 2*gamma_epsilon*Delta_gamma;
    alpha =max(max(abs(omega(:))),2/epsilon^2);
    M = 1./(1 + dt*( 1*sigma*epsilon^2*Delta.^2  - Delta  +   alpha - beta*Delta));

    
    Delta_u = (ifftn(Delta.*fftn(u)));  mu = Delta_u - F_prim(u)/epsilon^2;
    Delta_Wu = (ifftn(Delta.*fftn(F_prim(u)/epsilon^2)));
    res = sigma*epsilon^2*Delta_Wu + sigma*F_seconde(u).*(mu) + alpha*u - beta*Delta_u - F_prim(u)/epsilon^2; 
    u = real(ifftn(M.*(fftn( u + dt*res + dt*omega)))); 
    u = min(max(u,0),0.25);
  
     
    
   
    i;
if mod(i,30)==1
Ge1 = compute_geodesic_l_modif(U,Mreg,gamma10_1,gamma10_2,gamma10_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
Ge2 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
Ge3 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
Ge4 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
Ge =max(max(max(Ge1,Ge2),Ge3),Ge4);

end
% if mod(i,60)==11
% 
% Ge1 = compute_geodesic_l_modif(U,Mreg,gamma10_1,gamma10_2,gamma10_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% Ge2 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% Ge3 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% Ge4 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% % Ge5 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% % Ge6 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma6_1,gamma6_2,gamma6_3,epsilon);
% 
% Ge =max(max(max(Ge1,Ge2),Ge3),Ge4);
%  end 
% 
% if mod(i,60)==21
% 
% Ge1 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% Ge2 = compute_geodesic_l_modif(U,Mreg,gamma10_1,gamma10_2,gamma10_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% Ge3 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% Ge4 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% % Ge5 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% % Ge6 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma6_1,gamma6_2,gamma6_3,epsilon);
% 
% Ge =max(max(max(Ge1,Ge2),Ge3),Ge4);
% end 
% 
% 
%      if mod(i,60)==31 
% 
% Ge1 = compute_geodesic_l_modif(U,Mreg,gamma15_1,gamma15_2,gamma15_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% Ge2 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% Ge3 = compute_geodesic_l_modif(U,Mreg,gamma13_1,gamma13_2,gamma13_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% Ge4 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% % Ge5 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% % Ge6 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma6_1,gamma6_2,gamma6_3,epsilon);
% 
% 
% Ge = max(max(max(Ge1,Ge2),Ge3),Ge4);
% 
% 
%      end
% 
%    if mod(i,60)==41
% Ge1 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% Ge2 = compute_geodesic_l_modif(U,Mreg,gamma15_1,gamma15_2,gamma15_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% Ge3 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% Ge4 = compute_geodesic_l_modif(U,Mreg,gamma13_1,gamma13_2,gamma13_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% % Ge5 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% % Ge6 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma6_1,gamma6_2,gamma6_3,epsilon);
% 
% 
% Ge = max(max(max(Ge1,Ge2),Ge3),Ge4);
% 
% 
%   end
% % 
% %    if mod(i,60)==31
% % Ge1 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% % Ge2 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% % Ge3 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% % Ge4 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% % Ge5 = compute_geodesic_l_modif(U,Mreg,gamma15_1,gamma15_2,gamma15_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% % Ge6 = compute_geodesic_l_modif(U,Mreg,gamma10_1,gamma10_2,gamma10_3,gamma6_1,gamma6_2,gamma6_3,epsilon);
% % 
% % 
% % Ge = max(max(max(max(max(Ge1,Ge2),Ge3),Ge4),Ge5),Ge6);
% % 
% %   end
% % 
% %  if mod(i,60)==41
% % Ge1 = compute_geodesic_l_modif(U,Mreg,gamma15_1,gamma15_2,gamma15_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% % Ge2 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% % Ge3 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% % Ge4 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% % Ge5 = compute_geodesic_l_modif(U,Mreg,gamma13_1,gamma13_2,gamma13_3,gamma5_1,gamma5_2,gamma5_3,epsilon);
% % Ge6 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma6_1,gamma6_2,gamma6_3,epsilon);
% % 
% % 
% % Ge = max(max(max(max(max(Ge1,Ge2),Ge3),Ge4),Ge5),Ge6);
% % 
% %  
% %   end
% 
% %   if mod(i,60)==51
% % 
% % Ge1 = compute_geodesic_l_modif(U,Mreg,gamma14_1,gamma14_2,gamma14_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% % Ge2 = compute_geodesic_l_modif(U,Mreg,gamma9_1,gamma9_2,gamma9_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% % Ge3 = compute_geodesic_l_modif(U,Mreg,gamma15_1,gamma15_2,gamma15_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% % Ge4 = compute_geodesic_l_modif(U,Mreg,gamma7_1,gamma7_2,gamma7_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% %  
% % Ge = max(max(max(Ge1,Ge2),Ge3),Ge4);
% % 
% % 
% %   end
% % 
% %    if mod(i,60)==1
% % 
% % Ge1 = compute_geodesic_l_modif(U,Mreg,gamma15_1,gamma15_2,gamma15_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
% % Ge2 = compute_geodesic_l_modif(U,Mreg,gamma12_1,gamma12_2,gamma12_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
% % Ge3 = compute_geodesic_l_modif(U,Mreg,gamma8_1,gamma8_2,gamma8_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
% % Ge4 = compute_geodesic_l_modif(U,Mreg,gamma13_1,gamma13_2,gamma13_3,gamma4_1,gamma4_2,gamma4_3,epsilon);
% %  
% % Ge = max(max(max(Ge1,Ge2),Ge3),Ge4);
% % 
% % 
% %   end

    if mod(i,100)==1 
    
        
     clf;
    
 v = 4*u;
 alpha(1)
 p = patch(isosurface(x,x,x,v,0.75));
 isonormals(x,x,x,v,p)
 set(p,'FaceColor','blue','EdgeColor','none');

 
  
        w=real(X1_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');


   w=real(X2_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');



   w=real(X3_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');

   w=real(X4_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');

%   w=real(X5_tube);
%   p2 = patch(isosurface(x,x,x,w,0.25,w));
%   isonormals(x,x,x,w,p2)
%   set(p2,'FaceColor','green','EdgeColor','none');
% 
%   w=real(X6_tube);
%   p2 = patch(isosurface(x,x,x,w,0.25,w));
%   isonormals(x,x,x,w,p2)
%   set(p2,'FaceColor','green','EdgeColor','none');


  w=real(X7_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


  w=real(X8_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');

 w=real(X9_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X10_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X12_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X13_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X14_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');

 w=real(X15_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');



daspect([1 1 1])
 view([1,0.5,1]); 
 camlight headlight;
%camlight left;
%camlight(2,2)
 %camlight('infinite')
lighting gouraud


         view(-250,30);
     axis([0,1,0,1,0,1])
     pause(0.1)




frame = getframe(gcf);
 writeVideo(vv,frame);
       
    
    end


if (i*dt>T_vec(j_vec))
    
        clf;
    
 v = 4*u;
 alpha(1)
 p = patch(isosurface(x,x,x,v,0.9));
 isonormals(x,x,x,v,p)
 set(p,'FaceColor','blue','EdgeColor','none');

 
  
     
        w=real(X1_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','Green','EdgeColor','none');


   w=real(X2_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');



   w=real(X3_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');

  w=real(X4_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');

%   w=real(X5_tube);
%   p2 = patch(isosurface(x,x,x,w,0.25,w));
%   isonormals(x,x,x,w,p2)
%   set(p2,'FaceColor','green','EdgeColor','none');
% 
%   w=real(X6_tube);
%   p2 = patch(isosurface(x,x,x,w,0.25,w));
%   isonormals(x,x,x,w,p2)
%   set(p2,'FaceColor','green','EdgeColor','none');


  w=real(X7_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


  w=real(X8_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');

w=real(X9_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X10_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X12_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X13_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');


   w=real(X14_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');

 w=real(X15_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','red','EdgeColor','none');





daspect([1 1 1])
 view([1,0.5,1]); 
 camlight headlight;
%camlight left;
%camlight(2,2)
 %camlight('infinite')
lighting gouraud


       
        view(-250,30);
    axis([0,1,0,1,0,1])
     pause(0.1)



    
    title(['n = ',num2str(i)]);
    name_fig = ['Test_plateau_simple_cube_',num2str(j_vec),'.eps']; 
    print('-depsc', name_fig)
       j_vec = j_vec +1;

end



    
 
       
    
      
end
 

