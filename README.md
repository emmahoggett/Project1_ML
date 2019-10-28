# EPFL Machine Learning Higgs 2019

### Description
EPFL Machine Learning Higgs is a project that was launched in 2014 by the [CERN](https://home.cern/news/news/computing/higgs-boson-machine-learning-challenge) and is aimed to determine the presence of a Higgs Boson with measurements. To do so, a model has to be defined through machine learning methods, with a supplied training data.

After the model is determined with the training data, the results are submitted on the [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019) platform, in order to measure the accuracy of the model's predictions.

### Getting Started
This version was designed for python 3.6.6 or higher. To run the model's calculation, it is only needed to execute the file `run.py`. On the terminal, the command is `python run.py`. The code should return a `.csv` file with all its predictions, from the test data.


### Prerequisites
A train data and a testing data, where the results are unknown, are needed in `.csv` format. These data are available on the [Resources](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files) of AIcrowd platform.

To avoid dysfunctions, the code need the files:
* `implementation.py`
* `feature_expansion.py`
* `pre_processing.py`
* `proj1_helpers.py`

Those files contain functions that are used by the main code `run.py`.

### Additional content
The folder `Test_implementations` also contains the testing of the different functions implemented in `implementation.py`, under different conditions. To run those files, the same files as before are needed.

The file `ML_Projet1.pdf` correspond to the documentation of the project.

### Documentation
* [Class Project 1](https://github.com/epfml/ML_course/raw/master/projects/project1/project1_description.pdf) : Description of the project.
* [Resources](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files): Datas for the training and testing.

### Authors
* Group name: [MLBudget](https://www.aicrowd.com/teams/ML_Budget)
* Members: Aubet Louise, Cadillon Alexandre, Hoggett Emma

### Project Status
The project was submitted on the 28 October 2019, as part of the [Machine Learning](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) course.
