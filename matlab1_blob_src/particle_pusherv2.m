% clear all;

%% PIC PUSHER
dircur =pwd;

%% initialize the push procedure
%% time stepclc
ndt =100;
dt = sqrt((x0_map(end)-x0_map(1))^2+(y0_map(end)-y0_map(1))^2+(z0_map(end)-z0_map(1))^2)/(norm(Vxyz_entrance(:,1),2)*1e6)/ndt;

% physical constant
c = 2.998*10^8;
e = 1.602*10^-19;
Mp = 1.673*10^-27;
g = 1/sqrt(1-norm(Vxyz_entrance(:,1),2).^2/c^2);
A = e*dt/(2*Mp*g);

%no=round(rand(1)* size(XYZ_entrance,2));
Pxyz = XYZ_entrance*1e-6; %m
Vxyz = Vxyz_entrance; % 

XYZ_exit = Pxyz*0;
Vxyz_exit = Vxyz*0;
it =0;
leave = 0;

%% loop
while (leave < size(XYZ_exit,2) && it < ndt*5)  
    it =it+1;
    % read Bfield
    iX = round((Pxyz(1,:)*1e6-x0_map(1))/res_map+1);
    iY = round((Pxyz(2,:)*1e6-y0_map(1))/res_map+1);
    iZ = round((Pxyz(3,:)*1e6-z0_map(1))/res_map+1);
    indice = sub2ind(size(mapx),iX,iY,iZ);
    Bx = mapx(indice);
    By = mapy(indice);
    Bz = mapz(indice);
    
    % equation Boris method + update values Pxyz, Vxyz
    t=A*[Bx;By;Bz];
    No = t(1,:).^2+t(2,:).^2+t(3,:).^2;
    s = 2*t./(1+repmat(No,3,1));
    Pxyz = Pxyz + Vxyz*dt/2;

    Vxyz = Vxyz + cross(Vxyz+cross(Vxyz,t),s);
    Pxyz = Pxyz + Vxyz*dt/2;
    % check if still in the box
    I1 = find(Pxyz(1,:)*1e6<x0_map(1) | Pxyz(1,:)*1e6>x0_map(end));
    I2 = find(Pxyz(2,:)*1e6<y0_map(1) | Pxyz(2,:)*1e6>y0_map(end));
    I3 = find(Pxyz(3,:)*1e6<z0_map(1) | Pxyz(3,:)*1e6>z0_map(end));
    I = union(union(I1,I2),I3);
    
    % store the ones that left the box and remove them from computation
    if (~isempty(I))
        XYZ_exit( :,(leave+1) : (leave+size(I,2)) ) =Pxyz(:,I)*1e6; %µm
        Vxyz_exit(:,(leave+1) : (leave+size(I,2)) ) =Vxyz(:,I); %m/s
        
        Pxyz(:,I) =[];
        Vxyz(:,I) =[];
        leave = leave+ size(I,2);
    end
%     XYZ(:,:,it) = Pxyz(:,:);
%     Bx_xyz(:,it) =Bx(:);
%     By_xyz(:,it) =By(:);
%     VXYZ(:,:,it) = Vxyz(:,:);
%     Fxyz (:,:,it)= cross(Vxyz,[Bx;By;zeros(size(Bx))]);
    disp(['Particle_out_this_turn:',num2str(size(I,2)),'_particle left:', num2str(size(XYZ_exit,2)-leave)]);
end
if (leave < size(XYZ_exit,2))
    XYZ_exit(:,leave+1:end)= [];
    Vxyz_exit(:,leave+1:end)= [];
end
% if not, remove from the iteration ...
%% end loop

% XYZfinal

% for i=1:100
% %     %plot(reshape(XYZ(1,i,:),1,[])*1e6,reshape(XYZ(2,i,:),1,[])*1e6,'r');hold on;
%     hold on; plot(reshape(XYZ(1,i,:),1,[])*1e6,reshape(Fxyz(1,51,:),1, []),'r',reshape(XYZ(1,i,:),1,[])*1e6,reshape(Fxyz(2,51,:),1, []),'b',reshape(XYZ(1,i,:),1,[])*1e6,reshape(Fxyz(2,51,:),1, []),'g');hold on;
% end 
