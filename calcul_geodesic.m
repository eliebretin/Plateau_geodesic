function [Geodesic,Ge] = calcul_geodesic(D,x1),

[n1,n2] = size(D);
Ge = zeros(n1,n2);

Xn = x1;
Ge(Xn(1),Xn(2)) = 1;
%indice1 = [-1,0,1,1,1,0,-1,-1];
%indice2 = [-1,-1,-1,0,1,1,1,0];


indice1 = [-1,0,1,1,1,0,-1,-1,-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2];
indice2 = [-1,-1,-1,0,1,1,1,0,-2,-2,-2,-2,-2,-1,0,1,2,2,2,2,2,1,0,-1];


Geodesic = Xn;
while D(Xn(1),Xn(2))>0 && length(Geodesic)< n1*n2;
     
    
     for i=1:24,
        val_Di(i) = D(Xn(1)+ indice1(i),Xn(2)+ indice2(i)); 
     end         
     [val_min,indice_i] = min(val_Di);
    
     Xn(1) = Xn(1) +  indice1(indice_i);
     Xn(2) = Xn(2) +  indice2(indice_i);
     Geodesic = [Geodesic,Xn];
     %Ge(Xn(1),Xn(2)) = sqrt(indice1(indice_i).^2 + indice2(indice_i).^2).*(D(Xn(1) -  indice1(indice_i),Xn(2) -  indice2(indice_i)) + D(Xn(1),Xn(2)))/2;
     Ge(Xn(1),Xn(2)) = sqrt(indice1(indice_i).^2 + indice2(indice_i).^2);
     
     
end




end

