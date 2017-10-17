# Special course: Machine Learning and Simulation

## Collection of scripts from bachelor project 
### Scripts used when trying to solve Laplace 
1) lap3.py    : Main script (Python)
2) initParam  : Script used to get parameters used in Runge-Kutta loop (MatLab)
3) (excluding other scripts created by Allan; GlobalVariablesSEM2D, StartupLaplaceSEM2D, LinearStandingwave1D etc.)

### Example of data sets (e.g. use to check code)
1) laplaceTrainArchitectureNX61T500   : Training set with 61 nodes and 500 time step (csv file)
2) laplaceTestArchitectureNX61T500    : Test data with 61 nodes and 500 time step (used to check error) (csv file)

### Scripts used to create training data
1) DriverLaplaceSolverLinearStandingWavesInTimeAnalyticData   : Analytic Phi and analytic W (Matlab)
2) DriverLaplaceSolverLinearStandingWavesInTimeSEMData        : Analytic Phi and W from Laplace Solver (MatLab)
3) DriverLaplaceSolverLinearStandingWavesInTimeRKData         : Analytic inital phi then Runge-Kutta loop to determine phi and W (MatLab)
