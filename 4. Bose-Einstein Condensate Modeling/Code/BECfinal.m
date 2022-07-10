% close all; clear all; clc
function [psi1_f, psi2_f] = BECfinal()

A = -1*ones(3,1); B = -1.*A; % could set different values for individual parameters
L = 2*pi;%2*pi; % define computational domain [-L/2 L/2]
n = 16; % define number of Fourier modes 2^n

tmax = 4;
dt = 0.5;
% tmax = 0.5;
% dt = 0.1;
tspan = 0:dt:tmax;

x2 = linspace(-L/2,L/2,n+1);
x = x2(1:n); y = x; z = y;
[X, Y, Z] = meshgrid(x,y,z);

kx = (2*pi/L)*[0:(n/2-1) (-n/2):-1]; % rescale to 2pi domain on x
ky = kx; kz = ky;  % rescale to 2pi domain on y and z
[Kx, Ky, Kz] = meshgrid(kx,ky,kz);
Lap = (Kx.^2 + Ky.^2 + Kz.^2)/2;

% Compute forcing term in spatial domain 
C = (A(1)*(sin(X)).^2 + B(1)).*(A(2)*(sin(Y).^2) + B(2)).*(A(3)*(sin(Z).^2) + B(3));

% need shifted K terms for visualization
Kx_s = ifftshift(Kx);Ky_s = ifftshift(Ky);Kz_s = ifftshift(Kz);
Lk =max(max(max(Kz))); % find outer bound of k values (same for all directions)

% set up initial conditions
psi0_t1 = cos(X).*cos(Y).*cos(Z);
psi0_t2 = sin(X).*sin(Y).*sin(Z);
psi0_f1 = reshape((fftn(psi0_t1)),n^3,1);
psi0_f2 = reshape((fftn(psi0_t2)),n^3,1);

% %%  Evaluate system with ode45
opts = odeset('AbsTol',1e-6,'RelTol',1e-6);

[~, psi1_f] = ode45(@(t,y) rhsNLSE(t,y,Lap,C,n),tspan,psi0_f1,opts);
[~, psi2_f] = ode45(@(t,y) rhsNLSE(t,y,Lap,C,n),tspan,psi0_f2,opts);

m = length(tspan);
% [m ~] = size(psi1_f); % get the number of time steps for unpacking
% [m ~] = size(psi2_f);

% Reshape psi for visualization
for j = 1:m
    psi1_t(:,:,:,j) = ifftn((reshape(psi1_f(j,:),[n,n,n])));
    psi1_fshift(:,:,:,j) = fftshift(reshape(psi1_f(j,:),[n,n,n]));
    psi2_t(:,:,:,j) = ifftn((reshape(psi2_f(j,:),[n,n,n])));
    psi2_fshift(:,:,:,j) = fftshift(reshape(psi2_f(j,:),[n,n,n]));   
end

% %% Plot solutions a and b
% tic 
% close all
% title1 =  sprintf('BECa_spatial_fourier_%0d_L%0dpi_t%0d_dt%1g_A%G_B%G_sym_1.gif',n,L/pi,tmax,dt,A(1),B(1));
% animate581_BEC4(psi1_t, psi1_fshift,X,Y,Z,Kx_s,Ky_s,Kz_s,n,tspan,L,Lk,title1,A,B);
% %
% % close all
% title5 =  sprintf('BECb_spatial_fourier_%0d_L%0dpi_t%0d_dt%1g_A%G_B%G_sym_1.gif',n,L/pi,tmax,dt,A(1),B(1));
% animate581_BEC4(psi2_t, psi2_fshift,X,Y,Z,Kx_s,Ky_s,Kz_s,n,tspan,L,Lk,title5,A,B);
% toc % Elapsed time is 18.149489 seconds.

end