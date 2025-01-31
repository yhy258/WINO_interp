
<h1> Wave Interpolation Neural Operator: Parameter-Efficient Electromagnetic Surrogate Solver for Broadband Field Prediction using Discrete Wavelength Data </h1>

<hr />

> **Abstract:** *Recently, electromagnetic surrogate solvers, trained on solutions of Maxwell’s equations under specific simulation conditions, enabled real-time inference of computationally expensive simulations. However, conventional surrogate solvers often consider only a narrow range of simulation parameters and fail when encountering even slight variations in simulation conditions. To address this limitation, we propose a broadband surrogate solver capable of providing solutions across a continuous range of wavelengths, even when trained on discrete wavelength data. Our approach leverages a Wave-Informed element-wise Multiplicative Encoding and a Fourier Group Convolutional Shuffling operator to mitigate overfitting while capturing the fundamental characteristics of Maxwell’s equations. Compared to the state-of-the-art surrogate solver, our model achieves a 74% reduction in parameters, an 80.5% improvement in prediction accuracy for untrained wavelengths, demonstrating superior generalization and efficiency.* 
<hr />
</div>

## Introduction

Conventional neural network based surrogate solvers typically perform well within the fixed simulation settings in which they are trained.
Generating a new dataset and retraining the model to predict outcomes for new simulation parameters
are necessary.

To address this issue, we propose a parameter-efficient surrogate solver named Wave Interpolation Neural Operator (WINO). For the first time, WINO enables the interpolation of simulation parameters by training with discrete wavelength simulation data, allowing it to infer across a continuous spectrum of broadband wavelengths.

## Results
<img src="figures/SupFig.png" width="500"/>

The above figure shows a comparison of WINO with various surrogate solvers in terms of the number of parameters and prediction performance for untrained wavelengths.

Compared to the state-of-the-art model (NeurOLight), we achieve a $74\%$ reduction in parameters and $80.5\%$ improvements in prediction accuracy for untrained wavelengths, and $13.2\%$ improvements for trained wavelengths.

## Environment

>- python==3.9
>- pytorch==2.1.1
>- pytorch-cuda==11.8
>- pyutils == 0.0.1. [pyutils](https://github.com/JeremieMelo/pyutility)

We include the previous version **pyutils** repository in this code because the recent version of **pyutils** is not installed well.  
For **pyutils** installation,
```bash
cd pyutility
pip3 install --editable .
```




Other libraries listed in requirements.txt


## Data preparation
WINO dataset used in the experiments can be accessed on [Google Drive](https://drive.google.com/file/d/1Zx8Uu6mPba6uMvwkG0farp-AJ1j93gtt/view?usp=share_link).

The dataset should be placed in the directory defined in the configure file (check the configs directory).

The default path is "dataset/data/120nmdata"

```
WINO_interp (repository)
└───dataset
      └───data
            └───120nmdata
                ├───ceviche_train
                ├───ceviche_test
                └───ceviche_valid
```


### Simulation
However, you can also directly simulate the FDFD simulator. In this case, the dataset elements will be slightly different.

```bash
python CevicheSim/simulation.py
```
If you prepare the dataset using this simulation, the dataset will be created automatically in the default path.


## Training
1. Set the hyperparameter in configure yaml files.
2. Edit `train.py` code.

    "main(_model_)" : set the model name (yaml file name. examples: neurolight, myfno)
3. Conduct training.
    ```bash
    python meep_train.py
    ```
4. The validation results are logged in WANDB.





## Evaluation

### Checkpoint preparation

The pretrained WINO weights used in the paper can be accessed on [Google Drive](https://drive.google.com/file/d/1q6EYvwEC1bPmFMDf_JSXWOXer3eP41Bv/view?usp=drive_link).

The checkpoint should be placed in the directory defined in the configure file (check the configs directory).

The default path is "checkpoints/wino/nmse_waveprior_64dim_12layer_256_5060_group8"


```
WINO_interp (repository)
└───checkpoints
      └───wino
          └───nmse_waveprior_64dim_12layer_256_5060_group8
                ├───200epoch.pt
                └───best.pt
```
### Evaluation
1. You can change the model name of the evaluation target in `test/wvlwise_field_prediction.py`.

    ```python
    opt = test_yaml_to_args(parent_path=parent_path, model='wino')
    opt.model = 'wino'
    ```
2. Conduct evaluation
    ```bash
    cd test # "python" -> can be virtual env python file
    python wvlwise_field_prediction.py
    ```
3. Then, the result files are generated in test/test_wvlwise_results.
4. You can manually check the results through `test/test_wvlwise_results/test_field_prediction.ipynb`
