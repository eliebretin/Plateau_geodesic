close all;
clear all;
clf;


N = 2^7;
L = 1;
h = [1/N;1/N;1/N];
epsilon =2/N;


  vv = VideoWriter(['test_plateau_3curve1.avi'],'Motion JPEG AVI');
   vv.Quality = 90;
   open(vv);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Geodesic %%%%%%%%%%%%%ù
x = linspace(0,1,N);
[XX1,XX2,XX3] = meshgrid(x,x,x);
 R = 0.3;


gamma1_1f =  @(theta) 0.3*theta.^0 ;
gamma1_2f = @(theta)  1.2*R*sin(theta)+0.5 ;
gamma1_3f = @(theta)   1.2*R*cos(theta)+0.5 ;


gamma2_1f =  @(theta) 0.7*theta.^0 ;
gamma2_2f = @(theta)  1.2*R*sin(theta)+0.5 ;
gamma2_3f = @(theta)  1.2*R*cos(theta)+0.5 ;


gamma3_1f =  @(theta) 1.2*R*sin(theta)+0.5 ;
gamma3_2f = @(theta)  0.5*theta.^0  - 0.2*R*sin(theta);
gamma3_3f = @(theta) 0.7*R*cos(theta)+0.5;

 

%%%%%%%%%%%%%%%%%%%%%%% Pour l'affichage %%%%%%%%%%%%%%%%%%%%%%%%%%
Kb = 500; 
theta = linspace(0,2*pi,Kb+1);
theta = theta(1:Kb);

gamma2_1b = gamma2_1f(theta); gamma2_2b = gamma2_2f(theta); gamma2_3b = gamma2_3f(theta);
gamma1_1b = gamma1_1f(theta); gamma1_2b = gamma1_2f(theta); gamma1_3b = gamma1_3f(theta);
gamma3_1b = gamma3_1f(theta); gamma3_2b = gamma3_2f(theta); gamma3_3b = gamma3_3f(theta);
%%%%%%%%%%%%%%%%%%%%%%% Pour le calcul des géodésiques

K = 100; LK = K;
theta = linspace(0,2*pi,K+1); theta = theta(1:K);

gamma2_1 = gamma2_1f(theta); gamma2_2 = gamma2_2f(theta); gamma2_3 = gamma2_3f(theta);
gamma1_1 = gamma1_1f(theta); gamma1_2 = gamma1_2f(theta); gamma1_3 = gamma1_3f(theta);
gamma3_1 = gamma3_1f(theta); gamma3_2 = gamma3_2f(theta); gamma3_3 = gamma3_3f(theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

 X1_tube = zeros(N,N,N); 

 length_X1_tube = 0.3;
 Domaine1_tube =  zeros(N,N,N);

 
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
 for i=1:length(gamma2_2b)
 X3_tube = max(X3_tube,exp(-pi*length_X3_tube*((( gamma3_2b(1,i)-XX2).^2 + ( gamma3_1b(1,i)-XX1).^2 + ( gamma3_3b(1,i)-XX3).^2))/(epsilon^2)));
 end  



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k = [0:N/2,-N/2+1:-1]; [K1,K2,K3] = meshgrid(k,k,k); Delta = -4*pi^2*(K1.^2 + K2.^2 + K3.^2); 
Mreg = exp(+0.01*epsilon^2*Delta);
U = ones(N,N,N);
size(gamma1_1)
Ge1 = compute_geodesic_l_modif(U,Mreg,gamma2_1,gamma2_2,gamma2_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
Ge2 = compute_geodesic_l_modif(U,Mreg,gamma3_1,gamma3_2,gamma3_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
Ge3 = compute_geodesic_l_modif(U,Mreg,gamma1_1,gamma1_2,gamma1_3,gamma3_1,gamma3_2,gamma3_3,epsilon);


Ge = max(max(Ge1,Ge2),Ge3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u = 0.3*0.25*Ge/max(Ge(:));


F_prim = @(s) (1 - 6*s).*s;
F_seconde = @(s) (1 - 12*s);
dt =10*epsilon^2;
alpha =0/epsilon^2;
beta = 2/epsilon^0;
sigma =1;
M = 1./(1 + dt*( 1*sigma*epsilon^2*Delta.^2  - Delta  +   alpha - beta*Delta));
j_sauvegarde  = 1;

Mreg = exp(+0.1*epsilon^2*Delta);


T = 16
N_T = 10,
T_vec = linspace(0,T*(1.01),N_T);
j_vec = 1;




for i=1:T/dt,
gamma_epsilon =125/epsilon^3;

    
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
  




     if mod(i,50)==26 

  

Ge1 = compute_geodesic_l_modif(1-4*u,Mreg,gamma2_1,gamma2_2,gamma2_3,gamma1_1,gamma1_2,gamma1_3,epsilon);
Ge2 = compute_geodesic_l_modif(1-4*u,Mreg,gamma3_1,gamma3_2,gamma3_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
Ge3 = compute_geodesic_l_modif(1-4*u,Mreg,gamma1_1,gamma1_2,gamma1_3,gamma3_1,gamma3_2,gamma3_3,epsilon);


Ge = max(max(Ge1,Ge2),Ge3);

     end

  if mod(i,50)==1

  

Ge1 = compute_geodesic_l_modif(1-4*u,Mreg,gamma1_1,gamma1_2,gamma1_3,gamma2_1,gamma2_2,gamma2_3,epsilon);
Ge2 = compute_geodesic_l_modif(1-4*u,Mreg,gamma2_1,gamma2_2,gamma2_3,gamma3_1,gamma3_2,gamma3_3,epsilon);
Ge3 = compute_geodesic_l_modif(1-4*u,Mreg,gamma3_1,gamma3_2,gamma3_3,gamma1_1,gamma1_2,gamma1_3,epsilon);


Ge = max(max(Ge1,Ge2),Ge3);

     end




    
   
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
  set(p2,'FaceColor','Green','EdgeColor','none');


   w=real(X2_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');



   w=real(X3_tube);
  p2 = patch(isosurface(x,x,x,w,0.25,w));
  isonormals(x,x,x,w,p2)
  set(p2,'FaceColor','green','EdgeColor','none');


daspect([1 1 1])
 view([1,0.5,1]); 
 %camlight headlight;
%camlight left;
%camlight(2,2)
 camlight('infinite')
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
 p = patch(isosurface(x,x,x,v,0.75));
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


daspect([1 1 1])
 view([1,0.5,1]); 
 %camlight headlight;
%camlight left;
%camlight(2,2)
 camlight('infinite')
lighting gouraud


       
        view(-250,30);
    axis([0,1,0,1,0,1])
     pause(0.1)



    
    title(['n = ',num2str(i)]);
    name_fig = ['Test_plateau_3curve1_',num2str(j_vec),'.eps']; 
    print('-depsc', name_fig)
       j_vec = j_vec +1;

end



    
 
       
    
      
end
 

