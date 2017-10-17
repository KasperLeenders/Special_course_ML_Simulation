# Special_course_ML_Simulation

## Collection of scripts from bachelor project 
Scripts used when training and predicting
1) lap3.py    : Main script written in Python
2) initParam  : Script used to gete parameters used in Runge-Kutta loop written in MatLab

Example of data sets (e.g. use to check code)
1) laplaceTrainArchitectureNX61T500   : Training set with 61 nodes and 500 time step
2) laplaceTestArchitectureNX61T500    : Test data with 61 nodes and 500 time step (used to check error)

Scripts used to create training data
1) DriverLaplaceSolverLinearStandingWavesInTimeAnalyticData   : Analytic Phi and analytic W
2) DriverLaplaceSolverLinearStandingWavesInTimeSEMData        : Analytic Phi and W from Laplace Solver
3) DriverLaplaceSolverLinearStandingWavesInTimeRKData         : Analytic inital phi then Runge-Kutta loop to determine phi and W 
