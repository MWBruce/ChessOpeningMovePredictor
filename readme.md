# Chess Opening Move Predictor

This repository contains the code and resources for the Chess Opening Move Predictor project. Below is an overview of the repository structure:

## Repository Structure

- **data/**: Contains datasets used for training and testing the model.
    - `games.csv`: Dataset of chess openings and their moves.
    
   
- **src/**: Source code for the project.
    - `ChessOpeningMovePredictor.ipynb`: Notebook containing the experiments.
    - `scripts/DataLoading.py`: Script for preprocessing the dataset.
    - `scripts/Train.py`: Script defining the machine learning models, training and evaluation.
    - `scripts/Attention.py`: Script contianing Keras class for custom attention layer.
    - `archive/*`: Contains the experiments for the project.
    
- **requirements.txt**: List of dependencies required to run the project.

- **README.md**: This file, providing an overview of the repository.

## Getting Started

1. Clone the repository:
     ```sh
     git clone https://github.com/MWBruce/ChessOpenningMovePredictor.git
     ```
2. Install the required dependencies:
     ```sh
     pip install -r requirements.txt
     ```
3. Explore the data and train the models using the notebooks in the `src/` directory.