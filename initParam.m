function [E2, P2, Tmax, Twave, Np1D, Nk, rk4a, rk4b, rk4c, h, hx] = initParam(t)
% By Allan P. Engsig-Karup, apek@dtu.dk.
%
GlobalVariablesSEM2D; 

PL   = 3; % Pol. order in horizontal
Ps   = 4; % Pol. order in vertical, essentially nodes in vertical
NeX = 20; % No. of elements in the x-direction
NeY  = 1; % No. of elements in the y-direction
Nsteps = 100; % No. of time steps
Tmax = 20; % Number of wave periods in time to reach final time

disp(sprintf('Ps = %d, NeX = %d',Ps,NeX))

% Parameters
% Polynomial order used for approximation
N    = PL; % horizontal order
Ns   = Ps; % vertical order
vert = 1; % node clustering towards free surface on off

% Wave parameters
g      = 9.81;
Lx     = 1;
kh     = 1;
lwave  = Lx;
kwave  = 2*pi/lwave;
hd     = kh/kwave;
cwave = sqrt(g/kwave*tanh(kh));
Twave = lwave / cwave;
wwave = 2*pi / Twave;
Hwave = 0.02;

% Generate 2D Mesh
% adjust mesh to have element interfaces positioned near kinks in the
% bottom gradients
dxx = Lx / NeX;
x = [linspace(0,Lx,Lx/dxx+1)];
XX = x(:);
dxmin = min(diff(x));

% Setup all operators etc. on a given mesh
StartupLaplaceSEM2D;

% INITIAL CONDITION
% Determine analytical solution at free surface
time = t;
z = 0; % free surface level z=0
[U2, W2, E2, E2x, E2xx, P2] = LinearStandingwave1D(Hwave,cwave,kwave,z,hd,wwave,time,x1d);
h   = x1d*0+hd;
hx  = x1d*0;
%---------------------------------------------------------  
% Low storage Runge-Kutta coefficients
rk4a = [ 0.0   -567301805773.0/1357537059087.0  -2404267990393.0/2016746695238.0  -3550918686646.0/2091501179385.0  -1275806237668.0/842570457699.0];
rk4b = [ 1432997174477.0/9575080441755.0   5161836677717.0/13612068292357.0  1720146321549.0/2090206949498.0  3134564353537.0/4481467310338.0  2277821191437.0/14882151754819.0];			     
rk4c = [ 0.0  1432997174477.0/9575080441755.0  2526269341429.0/6820363962896.0  2006345519317.0/3224310063776.0  2802321613138.0/2924317926251.0 1.];
%---------------------------------------------------------
  