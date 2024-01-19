# UMvelocityVSresearch
This repository is for MATLAB code to train and test UM vertical velocity Virtual Sensors, researched by the Transport and Logistics Competence Centre of Vilnius Gediminas Technical University, Vilnius, Lithuania

Trained models for selected best window size for each type of artificial neural network (ANN) are in 'Trained models' folder.

Model training code with structures and parameters are in 'Model training' folder

There are validation and testing files for each time network in the repository's root. These files take trained networks and provide the same rmse_test result as in the published article.

There are launch files for the grid and Bayesian search combined to train, validate, and select the best structure for specified window sizes. It used a grid search for window size, for each window size uses Bayesian search to select the best ANN hyperparameters. It will output a Bayesian search report for each window size and all trained networks into separate folders for each window size.

Requirements to launch the code:
- Linux or Windows computer
- MATLAB at least R2019b with:
  - Statistics and Machine Learning Toolbox
  - Deep Learning Toolbox

Please reference our paper if you use our code for research:

To be published soon and will be here...

If you need any help contact the corresponding author of the research paper. We are interested in scientific or industrial collaboration in implementing Virtual Sensors into vehicles and any other systems.
