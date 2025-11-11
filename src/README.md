Folder Structure made off This: https://medium.com/@l.charteros/scalable-project-structure-for-machine-learning-projects-with-pytorch-and-pytorch-lightning-d5f1408d203e


TLDR:  
data: This directory is dedicated to storing all the data used in the project. It's further divided into two subdirectories:

raw: Contains the raw, untouched data as it was when collected or downloaded. Retaining the raw data separately is advantageous for those instances when reverting to the original data becomes necessary.
processed: Contains the data that has been processed and is ready to be used by the machine learning models. Usually, it contains the raw data split into train and test sets after some kind of preprocessing was applied. This can include cases where the data was cleaned, had features engineered, or was otherwise preprocessed.
.experiments: This directory stores the results of different model training runs. Each model may have several versions, each corresponding to a unique training run with potentially different hyperparameters or data. Although PyTorch Lighting names this directory .lightining_logs by default, it is preferred to rename it to make it easier to understand for anyone looking into the project for the first time.

src: This is the root directory for all the source code related to the project. Note that it also contains a subdirectory ml to further separate machine learning related code. This is especially beneficial when the machine learning module needs to integrate with another module (for instance, a backend), as it maintains a clean separation between different project modules, allowing team members to work on different parts simultaneously without interference.

ml/data: This directory holds scripts that handle data processing. It may include files like make_dataset.py (a script to download, filter, preprocess, and partition the raw data into training and test splits), and preprocessing.py (a script containing functions for data cleaning and preparation for modeling).

ml/datasets: This directory contains scripts that define how to load and handle the data used by the models. It might also contain subdirectories such as dataset1 for additional separation in case of multiple datasets. Each directory should contain at least two files:

dataset.py: Contains PyTorch's Dataset which allows for efficient and flexible data loading.
datamodule.py: ContainsPyTorch Lighting’s LightningDataModule which organizes the data loading and preparation steps and offers a clear and standardized interface for the data used in PyTorch Lightning systems.
ml/engines: This directory contains everything related to model training, validation, and testing and could also contain files related to these processes such as optimizers and schedules. For instance, system.py should include a LightningModule that defines the training, validation, and testing steps.

ml/models: This directory contains the scripts that define the different architectures of the models used in the project.

ml/scripts: This directory contains scripts for running different parts of the project, like train.py for training a model, test.py for testing, and predict.py for using a trained model to make predictions. These scripts typically include PyTorch Lightning’s Trainer class, which takes care of merging the LightningModule with the mLightningDataModule, running specified callbacks such as EarlyStopping and Checkpointing, and generally automating the entire machine learning process.

ml/utils: This directory contains helper functions, constants, and anything else that is used throughout the project. The general rule of thumb is that if an element is used across the project and doesn’t fit into the above directories, it should be placed in this directory.