
<h1> Wave Interpolation Neural Operator: Electromagnetic Surrogate Solver for Broadband Field Prediction using Discrete Wavelength Data </h1>

<hr />

> **Abstract:** *Deep learning based surrogate solvers promise fast electromagnetic simulations yet often fail when asked to predict fields at unseen wavelengths, which is crucial for broadband photonics. We present the principle of spectral consistency, which requires that any change in wavelength causes a predictable shift in spatial frequency, and embed it in a wave-informed, element-wise multiplicative encoding for the Fourier neural operator. Our model, trained only at 20 nm steps from 400 to 700 nm, accurately predicts full-domain electromagnetic fields in metalens, spectrum splitters, and freeform waveguides. The normalized mean squared error at unseen wavelengths decreases by up to 71\% relative to the current best neural solvers. The network uses only 0.43 million parameters, a 74\% reduction, and performs inference 42 times faster. The combination of physical insight and compact architecture yields a reliable surrogate solver for the rapid design of broadband next-generation photonic devices.* 
<hr />
</div>

## Introduction

Conventional neural network based surrogate solvers typically perform well within the fixed simulation settings in which they are trained.
Generating a new dataset and retraining the model to predict outcomes for new simulation parameters are necessary.

To address this issue, we propose a parameter-efficient surrogate solver named Wave Interpolation Neural Operator (WINO). For the first time, WINO enables the interpolation of simulation parameters by training with discrete wavelength simulation data, allowing it to infer across a continuous spectrum of broadband wavelengths.

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
WINO dataset used in the experiments can be accessed on   

Single-layer metalens: [Google Drive](https://drive.google.com/file/d/1Zx8Uu6mPba6uMvwkG0farp-AJ1j93gtt/view?usp=sharing).  
Multilyaer spectrum splitter: [Google Drive](https://drive.google.com/file/d/10Lj7GIuXQCZB199NqjACuBcq1Mg99Kdw/view?usp=sharing).  
Freeform waveguide: [Google Drive](https://drive.google.com/file/d/1qps565HT7ioUu2UBkQybejyPU3pSueoY/view?usp=sharing).

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
This simulation code is for the single-layer metalens setting. We will update the additional simulation codes.  

```bash
python CevicheSim/simulation.py
```
If you prepare the dataset using this simulation, the dataset will be created automatically in the default path.


## Training
1. Set the hyperparameter in configure yaml files.
2. Edit `automated_train.py` code.
3. Conduct training.
    ```bash
    python automated_train.py
    ```
4. The validation results are logged in WANDB.





## Evaluation

### Checkpoint preparation

The pretrained WINO weights used in the paper can be accessed on [Google Drive](https://drive.google.com/drive/folders/1Ma63gEch00QJ-fXQTDoKZeS4u4TchEAm?usp=drive_link).

The checkpoint should be placed in the directory defined in the configure file (check the configs directory).

Example:
```
WINO_interp (repository)
└───checkpoints
      └───wino
          └───nmse_waveprior_64dim_12layer_256_5060_group8
                ├───200epoch.pt
                └───best.pt
```
### Evaluation
1. Conduct evaluation
    ```bash
    cd test # "python" -> can be virtual env python file
    python wvlwise_field_prediction.py
    ```
2. Then, the result files are generated in test/test_wvlwise_results.
3. You can manually check the results through `test/test_wvlwise_results/test_field_prediction.ipynb`
