function [centers,p] = periodic_ran(R,a,sp,dr,fig1)

% R = 6e-4;
% a = 1e-4/2;
% dr =0.75*(2*a);
% sp = (2*a);
% x1 = 0;
% y1 = 0;

i=0;
centers =[];
p= [];

for x =[-R:sp:R]
    for y =[-R:sp:R]
        X= x+(2*rand()-1)*dr; 
        Y= y+(2*rand()-1)*dr;
        if (X^2+Y^2<R^2)
            i=i+1;
            centers(:,i) = [X;Y];
            p(i) =mod(i,2);
        end
    end
end
I = find(p);
J =find(p ==0);

if(fig1)
    figure;
    plot(centers(1,I),centers(2,I),'r+',centers(1,J),centers(2,J),'bo'); hold on; axis equal
    viscircles (centers(:,I)',ones(size(I))*a,'EdgeColor','r');viscircles (centers(:,J)',ones(size(J))*a,'EdgeColor','b');
    viscircles ([0;0]',R,'EdgeColor','k');
end
return;