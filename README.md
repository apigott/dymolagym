# DymolaGym: Applying Reinforcement Learning to Modelica Models in Dymola

The *DymolaGym* toolbox is developed to use Reinforcement Learning (RL) algorithms with models developed in the Modelica language and compiled and simulated in the Dymola application. Each Modelica model (.mo file) can be easily adapted to an OpenAI Gym wrapper by editing the environment configuration file, and assigning a reward function in Python. Minimal changes may be required in the Modelica model to enable simulation via the Dymola API.

Please note that this toolbox is developed on top of and in addition to the ModelicaGym library which this repository is forked from. We thank the authors of the ModelicaGym library for their efforts to create the original ModelicaGym with excellent documentation. Note that the current setup of the DymolaGym environment definition is not completely interchangeable with ModeilcaGym so for now we recommend installing the two repositories seperately if you would like to compare.

## Installation
The majority of the package dependencies for DymolaGym can be installed via pip and/or conda. However, as Dymola is a proprietary software, the `dymola` package is not available via a package manager. It may be installed from source if you have a licensed Dymola installation. Installation is convered briefly in the documentation but for clarity will be repeated below with additional installation tips.

1. It is recommended to install all packages inside a Conda environment to keep dependencies from causing issues. The following workflow is recommended:
```bash
:<work_dir>$ git clone https://github.com/apigott/modelicagym/
:<work_dir>$ cd modelicagym
:<work_dir>$ conda create --name <myenv> python=3.8 --file requirements.txt
:<work_dir>$ conda activate <myenv>
```
You can also add the required packages at any time using `conda install --file requirements.txt`

2. Install Dymola and add the `dymola.exe` to `$PATH`. On Windows the typical installation directory is `C:/Program Files/Dymola_2021x/`. On Linux the typical installation directory is `usr/bin/lib/Dymola_2021x/`. (Check your installation to be sure.)

3. Check that `dymola` is available as a system command. In terminal you can check that the command 
```bash
:<work_dir>$ dymola
```
starts a Dymola interface.

4. If you are running your Python installation of DymolaGym without a package manager (not recommended!) this should enable import of the `dymola` package in Python. You may check this by running the following command in terminal:
```bash
:<work_dir>$ python
>>> import dymola
```
5. If you are running your Python installation of Dymola in a Conda environment (recommended!) you will need to add the `dymola` package by creating a `<dymola>.pth` file. Create a `.pth` file (any name is fine) in `path/to/myenv/Lib/site-packages` with the following as text: `path/to/Dymola_2021x/Modelica/Library/python_interface/dymola.egg`. The same method as in 3 should confirm the Dymola-Python API is working.

Hint: You can find the path for your Conda environment with `conda env list`, use this path in conjunction with `/Lib/site-packages` to find the site-packages. You can use `which dymola` to find the Dymola installation directory (i.e. `path/to/dymola`). In Windows the path is likely `C:\Program Files`. In Linux the Dymola .exe is installed in `\bin` by default, but extra files (i.e. the python_interface directory) will be installed in `\opt`.

## Getting Started
Each `modelicagym\example` directory contains a different Modelica model and environment. `cartpole` is most up to date with the current `config.json` format. You can create your own environment by copying the `empty` or `cartpole` directory and following the prompts below in "Your First Env".

## Your First Env

### Changes to the Modelica .mo file
A couple small changes will need to be made to enable stepped simulation in the Dymola API using the `simulateExtendedModel()` method

1. All action values (values that you intend to change) should be modified to parameter values. Helpful hint: if you have a working model (`YourModel.mo`) that takes input blocks (`Modelica.Blocks.Interfaces.RealInput`) consider making an outer model (`YourModel_RL.mo`) with a constant block (`Modelica.Blocks.Sources.Constant`) with value `u` fed into the input of `yourModel`, where `u` is a parameter (declared as `parameter Real u`).

2. `initial equation` blocks must be changed to `equations`. You may access this by using the `when initial()` clause in the `equations` block. (This is a bug in the Dymola-Python API).

### Adding your model to the `modelicagym` package
When adding a new module to the `modelicagym` package it is helpful to understand how modules are imported in Python scripts. A recap is added below for beginner Python users:

* Calling `import <package>` imports all scripts included in `<package>/__init__.py` and runs all the Python code included in `<package>/__init__.py` and `<package>/setup.py`.
* If new modules are called in `<package>/__init__.py` (i.e. `from <package>.<subpackage> import *`) then everything listed in `<package>/<subpackage>/__init__.py` is imported and `<package>/<subpackage>/setup.py` is run. 
* The naming convention must be consistent, as everything else in Python.

To include your new package, modify the following files in the following ways:
  1. `modelicagym/examples/yourEnv/env.py` with your new environment class. For example you may add post or pre-processing of the states in the `.step()` method. Remember to call the parent class method (i.e. `return super().step(action)`) 
  
  2. `modelicagym/examples/yourEnv/__init__.py` with your prefered registered name in Gym. It should be in the format "<YourEnv>-v<versionNumer>"
  
  3. `modelicagym/examples/yourEnv/__init__.py` with the entry point (location) of the new class you have created. It should be in the format "modelicagym.examples:<YourEnv>"
  
  4. `modelicagym/examples/__init__.py` with an import of your environment (`from .yourEnv import *`)
  
### Running your new model
You can test whether or not all the necessary methods are working by using the `test.ipynb` or similar. The `test.ipynb` files also includes a very small example of training an agent using stable-baselines3 although in general you will need to train your agent for many more timesteps before seeing good results.
