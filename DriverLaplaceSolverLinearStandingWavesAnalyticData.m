close all
clear all
clc
%
% Driver script for executing a Laplace solver in 2D discretized
% using the Spectral Element Method (SEM).
%
% By Allan P. Engsig-Karup, apek@dtu.dk.
%
GlobalVariablesSEM2D; 

PL   = 3; % Pol. order in horizontal
Ps   = 4; % Pol. order in vertical, essentially nodes in vertical
NeX  = 70; % No. of elements in the x-direction
NeY  = 1; % No. of elements in the y-direction

disp(sprintf('Ps = %d, NeX = %d',Ps,NeX))

%% Parameters
% Polynomial order used for approximation
N    = PL; % horizontal order
Ns   = Ps; % vertical order
vert = 1; % node clustering towards free surface on off

%% Wave parameters
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

%% Generate 2D Mesh
% adjust mesh to have element interfaces positioned near kinks in the
% bottom gradients
dxx = Lx / NeX;
x = [linspace(0,Lx,Lx/dxx+1)];
XX = x(:);
dxmin = min(diff(x));

% Setup all operators etc. on a given mesh
StartupLaplaceSEM2D;

%% INITIAL CONDITION
% Determine analytical solution at free surface
% Create parameters used to set up training data

% No of nodes in x-direction
nn = length(x1d);
% No of time steps
tt = 1000;
% Number of periods
Tmax = 1;

% Allocate space 
U2 = zeros(nn,tt);
W2 = zeros(nn,tt);
W = zeros(nn,tt);
%E2 = zeros(31,6);
%E2x = zeros(31,6);
%E2xx = zeros(31,6);
P2 = zeros(nn,tt);

% Time step array
time = linspace(0,Tmax*Twave,tt);%0.25;
for i=1:tt
%for j=1:nn
%time(i)
z = 0; % free surface level z=0
[U2(:,i), W2(:,i), E2, E2x, E2xx, P2(:,i)] = LinearStandingwave1D(Hwave,cwave,kwave,z,hd,wwave,time(i),x1d);
            
% define bottom
h   = x1d*0+hd;
hx  = x1d*0;

% Initial conditions
E   = E2*0;
P   = P2(:,i);
%P = eye(nn);
%W(:,i) = E*0;
Ex  = E*0;

% Allocate variables for linear system
b   = X2D(:)*0;
sol = b*0;

% Laplace solver
%[W(:,i)] = LaplaceSEM2D(E,P);
% Exact solution
[U2(:,i), W2(:,i), E2, E2x, E2xx, P2(:,i)] = LinearStandingwave1D(Hwave,cwave,kwave,z,hd,wwave,time(i),x1d);
% Plot vertical velocity components for exact and computer solutions
%plot(x1d,W2,'k',x1d,W,'rx')
end
%% Create data from LaplaceSolver
data = [P2 W2 x1d];
csvwrite('laplaceTrainAnalyticNX70T1K.csv',data);