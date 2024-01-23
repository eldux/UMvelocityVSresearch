# Artificial Neural Networks for Real-World Data-Driven Virtual Sensors in Vehicle Suspension for UM vertical velocity estimation

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

E. Šabanovič, P. Kojis, V. Ivanov, M. Dhaens and V. Skrickij, "Development and Evaluation of Artificial Neural Networks for Real-World Data-Driven Virtual Sensors in Vehicle Suspension," in IEEE Access, doi: 10.1109/ACCESS.2024.3356715.

If you need any help contact the corresponding author of the research paper. We are interested in scientific and industrial collaboration in implementing Virtual Sensors into vehicles and any other systems.
