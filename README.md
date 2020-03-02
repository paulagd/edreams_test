# edreams_test

A github repository in order to do the test of EDREAMS.

### Requirements

* Python3
* Pytorch
* Jupyter notebook

### Instalation

> pip install -r venv.txt

### Usage

* __Python notebook__: Summary of everything done. Pipeline of exploring the data thus preprocessing it
 and balancing it. Also the model which allows for better results has been incorporated.

* __Source code__: Contains the code of the notebook unless the parts for visualizing data. However,
 it contains the parts for optimizing the models and for writing the .csv of the testing data 
 (also the testing data is processed here).
 
 
To run the code:

* Deep Learning model (Neural Network)

    ```
    > python main.py --DL
    ```
* LightGBM Classifier model

    - For optimizing it:
        ```
        > python main.py --optimize
        ```
        
    - For running the best model:
        ```
        > python main.py
        ```
        
> More flags existing in parse_args() function in case you want to try other combinations of preprocessing.
 


