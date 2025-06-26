close all;
clear all;
clf;


N = 2^6;
L = 1;
h = [1/N;1/N;1/N];
epsilon =1/N;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Geodesic %%%%%%%%%%%%%ù
x = linspace(0,1,N);
[XX1,XX2,XX3] = meshgrid(x,x,x);

Kb = 500; 
theta = linspace(0,2*pi,Kb+1);
theta = theta(1:Kb); R = 0.4;
% 

% 

gamma01_1b_f =  @(theta) 1*R*sin(theta)+0.5 ;
gamma01_2b_f = @(theta) 1*R*cos(theta)+0.5;
gamma01_3b_f = @(theta)  0.5*theta.^0 + ( 0.2*cos(4*theta) + 0.1*sin(theta));


gamma0_1b = gamma01_1b_f(theta(1)); gamma0_2b = gamma01_2b_f(theta(1)); gamma01_3b = gamma01_3b_f(theta(1));
gamma01_1b = gamma01_1b_f(theta); gamma01_2b = gamma01_2b_f(theta); gamma01_3b = gamma01_3b_f(theta);


K = 100; LK = K;

theta = linspace(0,2*pi,K+1); theta = theta(1:K);

gamma0_1 = gamma01_1b_f(theta(1)); gamma0_2 = gamma01_2b_f(theta(1)); gamma0_3 = gamma01_3b_f(theta(1));
gamma1_1 = gamma01_1b_f(theta); gamma1_2 = gamma01_2b_f(theta); gamma1_3 = gamma01_3b_f(theta);

    

 X1_tube = zeros(N,N,N);
 length_X1_tube = 0.2;
 Domaine1_tube =  zeros(N,N,N);

 
 for i=1:Kb
 X1_tube = max(X1_tube,exp(-pi*length_X1_tube*((( gamma01_2b(1,i)-XX2).^2 + ( gamma01_1b(1,i)-XX1).^2 + ( gamma01_3b(1,i)-XX3).^2))/(epsilon^2)));
 end  

k = [0:N/2,-N/2+1:-1]; [K1,K2,K3] = meshgrid(k,k,k); Delta = -4*pi^2*(K1.^2 + K2.^2 + K3.^2); 
Mreg = exp(+0.01*epsilon^2*Delta);
U = ones(N,N,N);

size(gamma1_1)
Ge = compute_geodesic_l_modif(U,Mreg,gamma0_1,gamma0_2,gamma0_3,gamma1_1,gamma1_2,gamma1_3,epsilon);

Mreg = exp(+0.1*epsilon^2*Delta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u = 0.25*Ge/max(Ge(:));

F_prim = @(s) (1 - 6*s).*s;
F_seconde = @(s) (1 - 12*s);


affiche_solution_3d2(x,4*u,0*u);
view(-100,20);
axis([-0,1,-0,1,-0,1])



dt =10*epsilon^2;
alpha =0/epsilon^2;
beta = 2/epsilon^0;
sigma =1.5;
M = 1./(1 + dt*( 1*sigma*epsilon^2*Delta.^2  - Delta  +   alpha - beta*Delta));
j_sauvegarde  = 1;




for i=1:50*100,
gamma_epsilon =200/epsilon^3;

    
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
  
     
    
   
    i
  




     if mod(i,50)==1 

  

   Ge = compute_geodesic_l_modif(1-4*u,Mreg,gamma0_1,gamma0_2,gamma0_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
   
     end

    
   
    if mod(i,100)==1 
    
        
     clf;
    


       affiche_solution_3d2(x,4*u,0*u);

  
        w=real(X1_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','cyan','EdgeColor','none');

       
       view(-140,20);
     axis([-0,1,-0,1,-0,1])
     pause(0.1)

%frame = getframe(gcf);
% writeVideo(vv,frame);
       
    
    end
    
 
       
    
      
end
 

