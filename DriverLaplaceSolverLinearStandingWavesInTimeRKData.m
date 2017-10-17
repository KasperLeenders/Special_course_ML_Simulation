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
Nsteps = 1000; % No. of time steps
Tmax = 1; % Number of wave periods in time to reach final time

%disp(sprintf('Ps = %d, NeX = %d',Ps,NeX))

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
time = 0.0;
z = 0; % free surface level z=0
[U2, W2, E2, E2x, E2xx, P2] = LinearStandingwave1D(Hwave,cwave,kwave,z,hd,wwave,time,x1d);
            
% define bottom
h   = x1d*0+hd;
hx  = x1d*0;

% Initial conditions
E   = E2;
P   = P2;
W   = E*0;
Ex  = E*0;

% Allocate variables for linear system
b   = X2D(:)*0;
sol = b*0;

% Laplace solver
[W] = LaplaceSEM2D(E,P);
% Exact solution
[U2, W2, E2, E2x, E2xx, P2] = LinearStandingwave1D(Hwave,cwave,kwave,z,hd,wwave,time,x1d);
% Plot vertical velocity components for exact and computer solutions
plot(x1d,W2,'k',x1d,W,'rx')

%%
% Initial conditions
E   = E2;
P   = P2;
W   = E*0;
Ex  = E*0;
Exx = E*0;
Px  = E*0;

%dt = 0.01; %Cr*dxmin/cwave
dt = (Twave*Tmax)/1000;

Nsteps = round(Tmax*Twave/dt);
dt = Tmax*Twave/Nsteps;

% Runge-Kutta residual storage
resE = zeros(length(E(:)),1);
resP = zeros(length(E(:)),1);

tmp = zeros(Np1D,Nk);

%---------------------------------------------------------  
% Low storage Runge-Kutta coefficients
rk4a = [ 0.0   -567301805773.0/1357537059087.0  -2404267990393.0/2016746695238.0  -3550918686646.0/2091501179385.0  -1275806237668.0/842570457699.0];
rk4b = [ 1432997174477.0/9575080441755.0   5161836677717.0/13612068292357.0  1720146321549.0/2090206949498.0  3134564353537.0/4481467310338.0  2277821191437.0/14882151754819.0];			     
rk4c = [ 0.0  1432997174477.0/9575080441755.0  2526269341429.0/6820363962896.0  2006345519317.0/3224310063776.0  2802321613138.0/2924317926251.0 1.];
%---------------------------------------------------------

close all
disp('Simulation started...')

b = X2D(:)*0;
sol = b*0;
tc = 0;

steptime = 10*Nsteps;

IIe = eye(N+1,N+1);
W0 = zeros(Nsteps,length(x1d));
P0 = zeros(Nsteps,length(x1d));

for tstep = 1:Nsteps
    disp(sprintf('Step %d/%d (Est. time remaining %.2f mins)',tstep,Nsteps,steptime/60*(Nsteps-tstep)))
    P0(tstep,:) = P;
    [W] = LaplaceSEM2D(E,P);
    W0(tstep,:) = W;
    % Runge-Kutta loop starts here
    tic;
    for INTRK = 1 : 5
        RKtime = time + dt*rk4c(INTRK);

        resE = rk4a(INTRK)*resE;
        resP = rk4a(INTRK)*resP;
        
        % Laplace solver
        [W] = LaplaceSEM2D(E,P);
        
        % Right hand side functions for ODE solver
        % Free surface equations (lineaerised, small-amplitude)
        rhsE = W;    % kinematic FS BC
        rhsP = -g*E; % dynamic FS BC
        
        resE = resE + dt*rhsE;
        resP = resP + dt*rhsP;

        % finish Runge-Kutta stage
        E      = E + rk4b(INTRK)*resE;
        P      = P + rk4b(INTRK)*resP;
        
        [U2, W2, E2, E2x, E2xx, P2] = LinearStandingwave1D(Hwave,cwave,kwave,z,hd,wwave,RKtime,x1d);
        %plot(x1d,E2,'k',x1d,E,'bx',x1d,W2,'k',x1d,W,'ro',x1d,P2,'k',x1d,P,'gs')
        %title(sprintf('time = %.2f',time))
        %xlabel('x')
        %axis([0 Lx -10*Hwave 10*Hwave])
        %drawnow
        %pause
    end
    steptime = toc;        
    
    % Runge-Kutta loop ends here
    if max(abs(E(:)+h(:)))<0 % dryland
        warning('Dry-land!')
        return
    end
   
    
    % update time
    time = time + dt;

end
disp('Simulation ended.')

%% Create data
data = [P0' W0' x1d];
csvwrite('laplaceTrainRKNX70T1K.csv',data);
return

