function [y0_map,x0_map,final_Az, scale,xyz,dxyz,sign] = bmap_generator(r, nblub, pix_size, number)
%% example
% clear all;
% close all;
dircur= pwd;

disp('program begins');

%% let's try blobs first
% r = 10;  % [1 2 4]; % microns
% nblub = 10; % [1 2 4]; % number of blobs
Bmax = 1; % [0.1 1 10]; % Tesla % not important
% pix_size = 0.1; % x r size of pixel in the box
margin = 7*r; % 10 different positions change the distance between them ==> packing ?

% pos = [?]; 

%% first define the blobs distribution

d_bcenter = 3; % x r
r_d_bcenter =0.2; % percent

%% initialization
xyz  = zeros(nblub,3); % position of the blobs
dxyz = zeros(nblub,3); % direction of the blobs
sign =zeros(nblub,1);  % orientation of the b-field

xyz(1,:)= rand(1,3)*d_bcenter*r;
sign(1) = 2*randi(2)-3;
theta = rand(2)*2*pi;
fi = acos(2*rand(2)-1);
dxyz(1,:) = [sqrt(1-cos(fi(2)).^2).*cos(theta(2)) , sqrt(1-cos(fi(2)).^2).*sin(theta(2)) , cos(fi(2))];


for n =2:nblub
    % neighbour
    xyz_n =xyz(randi(n-1),:);
    
    % new position and axis of the filament
    dist = d_bcenter*(1-(2*rand-1)*r_d_bcenter)*r;
    theta = rand(2)*2*pi;
    fi = acos(2*rand(2)-1);
    xyz(n,:)  = [sqrt(1-cos(fi(1)).^2).*cos(theta(1)) , sqrt(1-cos(fi(1)).^2).*sin(theta(1)) , cos(fi(1))]*dist+xyz_n;
    dxyz(n,:) = [sqrt(1-cos(fi(2)).^2).*cos(theta(2)) , sqrt(1-cos(fi(2)).^2).*sin(theta(2)) , cos(fi(2))];
    
    % sign of B-field ?
    sign(n) =  2*randi(2)-3;
    
end

disp('generate simulation box');

%% define simulation box edges, axis and box
scale = r*pix_size;

x_lim = [min(xyz(:,1))-margin,max(xyz(:,1))+margin];
y_lim = [min(xyz(:,2))-margin,max(xyz(:,2))+margin]; 
z_lim = [min(xyz(:,3))-margin,max(xyz(:,3))+margin];

x0_map = [x_lim(1):scale:x_lim(2)];
y0_map = [y_lim(1):scale:y_lim(2)];
z0_map = [z_lim(1):scale:z_lim(2)];

for i=1:size(z0_map,2)
    mapx{i} = sparse(zeros(size(x0_map,2),size(y0_map,2)));
    mapy{i} = sparse(zeros(size(x0_map,2),size(y0_map,2)));
    mapz{i} = sparse(zeros(size(x0_map,2),size(y0_map,2)));
    mapAz{i} = sparse(zeros(size(x0_map,2),size(y0_map,2)));
end

disp('generate blob base');
%% build blob base structure if doesn't exist

scale_base = r*pix_size/sqrt(3);
filename = ['base_blob_r_',num2str(r),'um_res_',num2str(scale_base),'um.mat'];
disp(filename);
f = dir('*.mat');
% if(~sum(ismember({f(:).name},filename)))
    x_lim_base = [-margin,margin]/sqrt(3);
    y_lim_base = [-margin,margin]/sqrt(3); 
    z_lim_base = [-margin,margin]/sqrt(3);

    x0 = [x_lim_base(1):scale_base:0,scale_base:scale_base:x_lim_base(2)];
    y0 = [y_lim_base(1):scale_base:0,scale_base:scale_base:y_lim_base(2)];
    z0 = [z_lim_base(1):scale_base:0,scale_base:scale_base:z_lim_base(2)];

    [X,Y,Z] = meshgrid(x0,y0,z0);
    rho= sqrt(X.^2+Y.^2);
    B0_norm= 1;
    Bx = -Y * B0_norm/r.*exp(-rho.^2/r^2-Z.^2/r^2);
    By = X * B0_norm/r.*exp(-rho.^2/r^2-Z.^2/r^2);
    Bz = Z.*0;
    Az = B0_norm*r/2.*exp(-rho.^2/r^2-Z.^2/r^2);
    
    Bx1 = sparse(reshape(Bx,1,[]));
    By1 = sparse(reshape(By,1,[]));
    Bz1 = sparse(reshape(Bz,1,[]));
    Az1 = sparse(reshape(Az,1,[]));
    X1 = sparse(reshape(X,1,[]));
    Y1 = sparse(reshape(Y,1,[]));
    Z1 = sparse(reshape(Z,1,[]));
%     save([dircur,'/',filename],'X1','Y1','Z1','Bx1','By1','Bz1','Az1');
% else
%     disp([filename,' is already computed']);
%     load([dircur,'/',filename])
%     X1
% end


disp('fill Bfield matrix');

%% input the blobs into the simulation box 

for n=1:nblub
disp(['blob #',num2str(n)]);
XYZpos = xyz(n,:);

%% 1) rotates blob base coordinates to match blob direction

u0 = dxyz(n,:); % blob direction

if(norm(u0(1:2)) == 0)
    disp('blob axis along z');
    X2 = X1;
    Y2 = Y1;
    Z2 = Z1;
    Bx2 = Bx1*Bmax*sign(n);
    By2 = By1*Bmax*sign(n);
    Bz2 = Bz1*Bmax*sign(n);
    Az2 = Az1*Bmax*sign(n);
else
    % rotation axis
    u_rot = [-u0(2);u0(1);0];
    % rotation angle
    theta = acos(u0(3)/norm(u0));
    c = cos(theta);
    s = sin(theta);
    
    % rotation matrix
    u_rot_n =u_rot./norm(u_rot,2);
    Ru = [u_rot_n(1)^2*(1-c)+c, u_rot_n(1)*u_rot_n(2)*(1-c)-u_rot_n(3)*s, u_rot_n(1)*u_rot_n(3)*(1-c)+u_rot_n(2)*s; ...
         u_rot_n(1)*u_rot_n(2)*(1-c)+u_rot_n(3)*s, u_rot_n(2)^2*(1-c)+c, u_rot_n(2)*u_rot_n(3)*(1-c)-u_rot_n(1)*s; ...
         u_rot_n(1)*u_rot_n(3)*(1-c)-u_rot_n(2)*s, u_rot_n(2)*u_rot_n(3)*(1-c)+u_rot_n(1)*s, u_rot_n(3)^2*(1-c)+c;];

    %% rotation
    X2 = Ru(1,:)*[X1;Y1;Z1];
    Y2 = Ru(2,:)*[X1;Y1;Z1];
    Z2 = Ru(3,:)*[X1;Y1;Z1];
    Bx2 = Ru(1,:)*[Bx1;By1;Bz1]*Bmax*sign(n);
    By2 = Ru(2,:)*[Bx1;By1;Bz1]*Bmax*sign(n);
    Bz2 = Ru(3,:)*[Bx1;By1;Bz1]*Bmax*sign(n);
    Az2 = Ru(3,3)*Az1*Bmax*sign(n); % Ax and Ay are null
end

%% Transfert to map

Ix = round((X2-x_lim(1)+xyz(n,1))/scale+1);
Iy = round((Y2-y_lim(1)+xyz(n,2))/scale+1);
Iz = round((Z2-z_lim(1)+xyz(n,3))/scale+1);

for i=1:size(z0_map,2)
    J = find(i == Iz);
    if(~isempty(J))
        [~,IA,~] = unique([Ix(J);Iy(J)]','rows'); % check no redondancy
        J = J(IA);
        mapx{i} =  mapx{i}+sparse(Ix(J),Iy(J),Bx2(J),size(mapx{i},1),size(mapx{i},2));
        mapy{i} =  mapy{i}+sparse(Ix(J),Iy(J),By2(J),size(mapx{i},1),size(mapx{i},2));
        mapz{i} =  mapz{i}+sparse(Ix(J),Iy(J),Bz2(J),size(mapx{i},1),size(mapx{i},2));
        mapAz{i} = mapAz{i}+sparse(Ix(J),Iy(J),Az2(J),size(mapx{i},1),size(mapx{i},2));
    end
end
end
%close(h);

%uisave({'mapx','mapy','mapz','x0_map','y0_map','z0_map','res_map'},[dircur,'/Bmap_B_',num2str(B0),'T_a_',num2str(a),'um_b_',num2str(b),'um_res_',num2str(res_map),'um.mat']);
% uisave({'mapx','mapy','mapz','mapAz','x0_map','y0_map','z0_map','scale','xyz','dxyz','sign'},[dircur,filename]);
name = sprintf('%s_%d','workspace',number);
save(name,'mapx','mapy','mapz','mapAz','x0_map','y0_map','z0_map','scale','xyz','dxyz','sign')
final_Az = zeros(size(mapAz{i}(:,:)));

for i=1:size(z0_map,2)
    final_Az = final_Az+full(mapAz{i}(:,:));
end
final_Az = final_Az*scale;
end
% figure;imagesc(y0_map,x0_map,final_Az*scale); axis tight equal;
% % hold on; plot(xyz(:,2),xyz(:,1),'r+')
% xlabel('y(microns)');
% ylabel('x(microns)');
% 
% N1= round(size(mapx,3)/2);
% N2= round(size(mapx,2)/2);
% N3= round(size(mapx,1)/2);
% figure; subplot(2,2,[1 3]);imagesc(y0_map,x0_map,mapx(:,:,N1)); axis equal
% subplot(2,2,2);imagesc(z0_map,x0_map,reshape(mapy(:,N2,:),size(x0_map,2),[])); axis equal
% subplot(2,2,4);imagesc(z0_map,x0_map,reshape(mapx(N3,:,:),size(y0_map,2),[])); axis equal
% %figure; subplot(1,3,1); contour(x0_map,y0_map,mapx(:,:,N),10);subplot(1,3,2); contour(x0_map,y0_map,mapy(:,:,N),10);subplot(1,3,3);contour(x0_map,y0_map,mapz(:,:,N),10)
%figure; contour(x0_map,y0_map,sqrt([mapx(:,:,N).^2+mapy(:,:,N).^2+mapz(:,:,N).^2]),10);%hold on;quiver(x0_map,y0_map,mapx(:,:,N),mapy(:,:,N));%subplot(1,3,2); quiver(x0_map,y0_map,mapy(:,:,N));subplot(1,3,3);quiver(x0_map,y0_map,mapz(:,:,N))


%B = interp3(nX,nY,nZ,nBx,-100:0.1:100,-100:0.1:100,-100:0.1:100);

