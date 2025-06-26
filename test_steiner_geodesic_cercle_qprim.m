clear all;
clf;
colormap('jet');



N = 2^7;
L = 1;
h = [1/N;1/N];


x = linspace(0,1,N);
[XX1,XX2] = meshgrid(x,x);
resG = 1;

M_d =4; %  nombre de poinds 


 
for m=1:M_d,
       X_p(1,m) = 1/2 + 0.4*cos(2*pi*((m-1)/M_d+sqrt(2)));
       X_p(2,m) = 1/2 + 0.4*sin(2*pi*((m-1)/M_d+sqrt(2)));
  end
   

 for m=1:M_d,
 X_pd(:,m) = 1 + round(X_p(:,m)*(N-1)); 
 end

 
k = [0:N/2,-N/2+1:-1];
[K1,K2] = meshgrid(k,k);
Delta = -4*pi^2*(abs(K1).^2 + abs(K2).^2);


F_prim = @(s) (1 - 6*s).*s;
F_seconde = @(s) (1 - 12*s);

epsilon =2/N;


dt =0.25*epsilon^2;
alpha =0/epsilon^2;
beta = 2/epsilon^0;
sigma =1;
M = 1./(1 + dt*( 1*sigma*epsilon^2*Delta.^2  - Delta  +   alpha - beta*Delta));

j_sauvegarde  = 1;


Mreg = exp(0.1*epsilon^3*Delta);


u = zeros(N,N);

options.nb_iter_max = Inf;





%%%%%%%%%%%%%% cas M_d = 3;



if (M_d == 2)
T = 0.04;
elseif (M_d == 3)
T = 0.1;
elseif (M_d == 4)
T = 0.1; 
elseif (M_d == 5)
T = 0.2;     
elseif (M_d == 6)
    T = 0.05
end


%%%%%%%%%%%%%% cas M_d = 4;

N_T = 10,
T_vec = linspace(0,T*(1.01),N_T);
j_vec = 1;

%%%%%%%%%%%%%%%%%%% initialisation %%%%%%
 W =  0.1 + (1-4*u);
 W = real(ifft2(Mreg.*fft2(W)));
   
  [D,S] = perform_fast_marching(1./W', X_pd(:,1), options);
    
  
   d_phi = 0;
   for m = 2:M_d,
    p1 = compute_geodesic(D, X_pd(:,m));
    Ge1 = zeros(size(D));
    for k=length(p1):-1:2,
        if norm(p1(:,k) - X_pd(:,1) )>1
        Ge1(round(p1(1,k)),round(p1(2,k))) =  Ge1(round(p1(1,k)),round(p1(2,k))) + norm(p1(:,k) - p1(:,k-1)); 
        end
    end
    Ge1 = real(ifft2(M.*fft2(Ge1)));
    
    d_phi =  max(d_phi,Ge1);
   end
 
 u = d_phi/max(d_phi(:))/4;  



for i=1:T/dt,
    
%%%%%%%%%%%%%% Minimization  \gamma %%%%%%%%%%  
  
   X_pd(:,[2:M_d,1]) = X_pd(:,1:M_d);
   W =  0.01 + (1-4*u);
   W = real(ifft2(Mreg.*fft2(W)));
   
  [D,S] = perform_fast_marching(1./W, X_pd(:,1), options);
    
  
   d_phi = 0;
   for m = 2:M_d,
    p1 = compute_geodesic(D, X_pd(:,m));
    Ge1 = zeros(size(D));
    for k=length(p1):-1:2,
        if norm(p1(:,k) - X_pd(:,1) )>1
        Ge1(round(p1(1,k)),round(p1(2,k))) =  Ge1(round(p1(1,k)),round(p1(2,k))) + norm(p1(:,k) - p1(:,k-1)); 
        end
    end
    Ge1 = real(ifft2(Mreg.*fft2(Ge1)));
    
    d_phi =  max(d_phi,Ge1);
   end
 



      
   
%%%%%%%%%%%%%% Minimization  U         %%%%%%%%%%%%%%%
    for k =1:10,
    
    gamma_epsilon =0.5/epsilon^2;
    Delta_gamma = d_phi.*(sqrt(1-4*u));
     
    omega = 2*gamma_epsilon*Delta_gamma;
     
    alpha =max(max(abs(omega(:))),2/epsilon^2);
    M = 1./(1 + dt*( 1*sigma*epsilon^2*Delta.^2  - Delta  +   alpha - beta*Delta));

   
    i
    
    Delta_u = (ifft2(Delta.*fft2(u)));  mu = Delta_u - F_prim(u)/epsilon^2;
    Delta_Wu = (ifft2(Delta.*fft2(F_prim(u)/epsilon^2)));
    res = sigma*epsilon^2*Delta_Wu + sigma*F_seconde(u).*(mu) + alpha*u - beta*Delta_u - F_prim(u)/epsilon^2; 
    u = real(ifft2(M.*(fft2( u + dt*res + dt*omega)))); 
    u = min(max(u,0),0.25);
  
    end


    
if (mod(i,10*M_d)==1)
    imagesc(x,x, u); 
     % colorbar;
      axis square;
       hold on;   
       for m=1:M_d 
    plot(X_p(2,m),X_p(1,m),'wO','linewidth',5)
       end     
      axis square;

%frame = getframe(gcf);
% writeVideo(vv,frame);

    pause(0.1);
end

if (i*dt>T_vec(j_vec))
    clf;
    imagesc(x,x, u); 
   
     hold on;   
       for m=1:M_d 
    plot(X_p(2,m),X_p(1,m),'wO','linewidth',5)
       end     
      axis square;
    
    
    title(['n = ',num2str(i)]);
    colorbar;
    axis square;
    %colormap('Hot')
    j_vec = j_vec +1;
  %  name_fig = ['Test_steiner_cercle_Md_',num2str(M_d),'method_qprim_t_',num2str(j_vec),'.eps']; 
 %print('-depsc', name_fig)

end


end

