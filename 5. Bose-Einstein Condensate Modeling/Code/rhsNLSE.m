function rhs = rhsNLSE(t,psi_fL,Lap,C,n)

% psi_fL: psi in  Fourier space as an n^3 x 1 vector
psi_f = reshape(psi_fL,[n,n,n]); % Reshape into n x n x n array

% Inverse FT for some operations in spatial domain
psi_t = ifftn((psi_f));      % Reshape into n^2 x 1 vector
psi_tL = reshape(psi_t,n^3,1);  % Reshape into n x n x n array


% psi n^3x1 vec in Fourier space
% Lpsi = reshape(Lap.*psi_fL/2,[n,n,n]); % doing el. ws. mult as vectors
Lpsi = Lap.*psi_f; % doing el. ws. mult as tensors

% Compute the non-linear terms separately, then combine
% (abs(psi_t).^2).*psi_t, (conj(psi_t).*psi_t).*psi_t
% NL1 = (reshape((abs(psi_fL).^2).*psi_fL,[n,n,n])); % compute multiplication after transforing
NL1 = (fftn(reshape((abs(psi_tL).^2).*psi_tL,[n,n,n])));

NL2 = (fftn(C.*psi_t)); % trig term
% NL2 = C_f.*psi_f; % trig term

rhs = reshape((-1i).*Lpsi + (-1i).*NL1 + (-1i).*NL2,n^3,1); % solve for psi
