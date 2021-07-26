# Deep Koopman operator learning - DKLc

## Requirements
The code is tested with Python Version 3.6.8-64 on Windows 10 and 3.6.9-64 in Ubuntu 18.04 and runs fastest with a cuda 
enabled graphics card with Cuda 10.1. If you want to use a gpu install the appropriate tensorflow-gpu package version 
1.14.

Install the required packages given in the requirements.txt.

## Manual
#### Experiments
An example file for the experiments can be found in the "experiments/*" folders.

#### Experiment Evaluation
The previously trained ANNs for the Van der Pol and three tank system are part of the repository. 
To obtain the predicted states execute the  "*Evaluation_1.py" files in the corresponding evaluation
folders.

To execute the evaluation files add "misc/Matlab" to MATLAB's path. 

## Acknowledgement
The code was partially written by my valuable colleague [Alexander Gräfe](https://www.dsme.rwth-aachen.de/cms/DSME/Das-Institut/Team-CMS-Artikel-/~melfa/Alexander-Graefe/ "Alexander Gräfe - Staff page"). Some parts of the code were taken from 
[Bethany Lusch](https://github.com/BethanyL/DeepKoopman "BetanyL GitHub").