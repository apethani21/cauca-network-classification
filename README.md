# Cauca Network Protocol Classification

## Data source

The dataset was obtained from [Kaggle](https://www.kaggle.com/jsrojas/ip-network-traffic-flows-labeled-with-87-apps).

## Exploration

Initial exploration is done in `explore_and_preprocess.ipynb`. Some things this flagged up and some steps taken:
- Needed to convert IP addresses into numerical features.
- Needed to encode the `Protocol` feature (TCP/UDP/Other).
- The raw dataset included 74 possible protocols. I keep the 50 most occuring protocols, removing protocols which occured at most 25 times in the dataset (in total this is approximately 0.008% of the data).
- The pandas-profiling report saved as `metadata_preprocessing/data_summary.html` flagged up the fact that many of the features are highly skewed.
- After this initial processing we're left with 3,567,193 rows and 73 features.
- In `yellowbrick_viz.ipynb` the Yellowbrick library is explored, and some basic tests show that a complex model is warranted.

## Establishing a baseline
After experimenting with various models, I decided to use a feedfoward neural network. In trying to establish a baseline using the `MLPClassifier` in scikit-learn, it became apparent that the choice of preprocessing steps made a significant difference in the final accuracy, for instance using PCA and a smaller layer led to drop in accuracy of 3.45% and applying only a log transformation to all features increased the accuracy by 3.39%. 

## Choosing preprocessing steps
The results in trying to establish a baseline led to the following plan and result:

Using a fixed, simple model, trial different combinations of preprocessing steps to find an optimum one before trying to explore various architectures for the neural network. This is done in `preprocessing_search.py`.

The preprocessing steps considered are whether to apply a log transformation, whether to apply PCA keeping enough components to include 97.5% of the explained variance, over-sampling techniques such as SMOTE, ADASYN and random over-sampling (using imblearn), and whether to add 15 custom features derived from the dataset.

As shown in `preprocessing_trials_analysis.ipynb` the final outcome was to include the custom features and apply a log transformation, but nothing further. The `feature_dive.ipynb` is where I tried to come up with potentially useful features.

## Searching for a better architecture

Here I chose to use Keras, Keras Tuner and Tensorboard to trial different architectures.
The train/validation sets form 60% and 20% of the data respectively, and 20% is held out as a test set, stratifying on the classes. This is done in `util.py`.

Trials were run using deep & wide, deep and narrow, and shallow neural network architectures. I then honed in on the deep and narrow neural networks which worked best, trying to tune the sizes of specific layers and find the best learning rate, after originally exploring learning rates that were too low.

## Result and evaluation

The [test_model](https://github.com/apethani21/cauca-network-classification/tree/main/test_model) folder contains the final model trained on the original train and validation sets combined, and evaluated on the test set. The final accuracy achieved on the test set was 77%.

The [final model architecture](https://github.com/apethani21/cauca-network-classification/blob/main/test_model/model_summary.txt) contains 4 deep, fully connected layers containing 256, 512, 769 and 512 nodes respectively, resulting in 967,730 trainable parameters.
