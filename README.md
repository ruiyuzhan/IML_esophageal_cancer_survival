# Building an Interpretable Machine Learning Prognosis Prediction Model Based on Baseline Examinations of Patients with Esophageal Cancer Undergoing Surgery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the MATLAB implementation for building an interpretable machine learning prognosis prediction model for esophageal cancer patients undergoing surgery, based on baseline examinations.


## Overview

This work presents an interpretable machine learning approach for predicting the prognosis of esophageal cancer patients who undergo surgery. The model utilizes baseline examination data to predict survival outcomes. The implementation uses Random Forest (TreeBagger) classifier with meta-heuristic optimization algorithms for hyperparameter tuning.

### Key Features

- **Multiple Optimization Algorithms**: Supports 30+ meta-heuristic optimization algorithms (PSO, GWO, SSA, DBO, etc.) for automatic hyperparameter optimization
- **Cross-Validation**: 10-fold cross-validation for robust model evaluation
- **Performance Metrics**: Comprehensive evaluation including ROC curves, DCA curves, AUC, accuracy, F1-score
- **Data Augmentation**: SMOTE-like synthetic data generation for handling class imbalance
- **Hyperparameter Ablation**: Systematic testing of different hyperparameter combinations

### Model Architecture

The model uses Random Forest (TreeBagger) with the following optimized hyperparameters:
- **NumTrees**: Number of trees in the ensemble (optimized range: 20-200)
- **MinLeafSize**: Minimum number of observations per tree leaf (optimized range: 2-10)

## Requirements / Environment Setup

### MATLAB Requirements

- **MATLAB R2018b or later**
- **Statistics and Machine Learning Toolbox** (required for TreeBagger)

### Installation Steps

1. **Check Toolbox Installation**

   Run the following script to verify that the required toolbox is installed:
   ```matlab
   cd scripts
   check_toolbox
   ```

2. **Set Up Data Paths**

   Configure the data paths in the configuration file:
   ```matlab
   cd scripts
   update_data_path
   ```

   This script will:
   - Check for the configuration file `R_11_Nov_2025_20_22_24.mat`
   - Update data paths to point to the correct data folder
   - Verify data file existence

3. **Verify Environment**

   The main script will automatically check for required toolboxes when executed. If the Statistics and Machine Learning Toolbox is not installed, you will be prompted to install it.

## Data Preparation

### Data Structure

The data is organized as follows:
```
data/
└── 原始数据.csv             # Original raw data
```


## Experiments

### Main Experiment: Model Training and Evaluation

**Script**: `scripts/train.m`

This is the main script for training and evaluating the prognosis prediction model.



**What it does**:
- Loads and preprocesses the data
- Splits data into training/validation/test sets (default: 70%/15%/15%)
- Performs data normalization (Z-score)
- Applies data augmentation (SMOTE-like)
- Optimizes hyperparameters using meta-heuristic algorithms
- Trains the Random Forest model
- Evaluates performance and plots ROC curves

**Key Parameters** :
- `random_seed`: Random seed for reproducibility (default: 42)
- `spilt_rio`: Data split ratio [train, validation, test] (default: [0.7, 0.15, 0.15])
- `num_pop1`: Population size for optimization (default: 30)
- `num_iter1`: Number of iterations for optimization (default: 50)
- `method_mti1`: Optimization algorithm name (default: 'PSO粒子群算法')
- `get_mutiple`: Data augmentation multiplier (default: 2)


### Optimization Algorithms

The code supports 30+ meta-heuristic optimization algorithms. Available algorithms include:

- **PSO** (Particle Swarm Optimization)
- **GWO** (Grey Wolf Optimizer)
- **SSA** (Sparrow Search Algorithm)
- **DBO** (Dung Beetle Optimizer)
- **SCA** (Sine Cosine Algorithm)
- **SA** (Simulated Annealing)
- **POA** (Pelican Optimization Algorithm)
- **AVOA** (African Vulture Optimization Algorithm)
- And many more...

To use a specific algorithm, set `method_mti1` in the configuration file to the algorithm name (e.g., 'PSO粒子群算法', 'GWO灰狼优化算法').

## File Structure

```
.
├── src/                          # Source code
│   ├── optimize_fitctreebag.m   # Hyperparameter optimization function
│   ├── RF_process.m              # Random Forest prediction processing
│   ├── generate_classdata.m     # Synthetic data generation
│   └── figure_data_generate.m   # Data visualization
├── scripts/                      # Execution scripts
│   ├── train.m                   # Main training script
│   ├── test_rf_hyperparameters.m      # Hyperparameter ablation script

├── data/                         # Data directory
│   └── 原始数据.csv            # Original raw data
├── LICENSE                      # MIT License
└── README.md                    
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the repository or contact the authors.

