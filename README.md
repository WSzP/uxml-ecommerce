# UXML for e-commerce
E-commerce recommender based on user events

## How to use this repo?
1. Start with *eda-uxml.ipynb* for exploratory data analysis
2. Prepare the data for machine learning with data *data-preparation-uxml.ipynb* (Optionally test the resulting *.npz* with *sparse-matrix-tester.ipynb*)
3. Run one of the training algorithms, such as *train-uxml-basic-matrix-factorization.ipynb* (more to come in the future)
4. Test the performance of the training alogrithm against a test set (different from train and validation data) with *test-uxml.ipynb* or *quick-test-uxml.ipynb*
5. Put the results in practice. Two use case examples are provided *use-case-examples.ipynb* (item recommender for users and minimalistic stock need prediction to help with e-commerce logistics)

## Automatic Hyper-parameter Optimization (HPO)
* The Optuna based implementation of HPO: *train-uxml-adam-optuna.ipynb*
* Applying the hyper-parameters found by Optuna without the need for Optuna framework: *train-uxml-adam-from-best-trial.ipynb*


## Notes
* The *data-preparation-test.ipynb* can be used to test the efficiency of data preparation, as in comparing the prepared data to the ground truth. This is not needed for the user-behaviour prediction process.
* The *quick-test-uxml.ipynb* relies on sparse matrix operations to do only MSE, RMSE, MAE, R-squared, and explained variance, for this reason it runs in 1.1s, compared to 175.89s of the full test which has multiple approaches, and many more metrics.

## Acknowledgments
* Machine Learning powered by PyTorchLightning [https://github.com/PyTorchLightning/pytorch-lightning]
* Exploratory Data Analysis powered by Pandas [https://github.com/pandas-dev/pandas]
* Data preparation powered by SciPy [https://github.com/scipy/scipy]
* Testing by Microsoft Best Practices on Recommendation Systems [https://github.com/microsoft/recommenders]
* Source of the data [https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop] (Thanks to REES46 Marketing Platform for this dataset.)


© Copyright 2020 Peter Szabo