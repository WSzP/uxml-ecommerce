# UXML for e-commerce
E-commerce recommender based on user events

## How to use this repo?
1. Start with *eda-uxml.ipynb* for exploratory data analysis
2. Prepare the data for machine learning with data *data-preparation-uxml.ipynb*
3. Optionally test the resulting *.npz* with *sparse-matrix-tester.ipynb*
4. Run one of the training algorithms, such as *train-uxml-basic-matrix-factorization.ipynb* (more to come in the future)
5. Test the performance of the training alogrithm against a test set (different from train and validation data) with *test-uxml.ipynb*

* The *data-preparation-test.ipynb* can be used to test the efficiency of data preparation, as in comparing the prepared data to the ground truth. This is not needed for the user-behaviour prediction process.

## Acknowledgments
* Machine Learning powered by PyTorchLightning [https://github.com/PyTorchLightning/pytorch-lightning]
* Exploratory Data Analysis powered by Pandas [https://github.com/pandas-dev/pandas]
* Data preparation powered by SciPy [https://github.com/scipy/scipy]
* Testing by Microsoft Best Practices on Recommendation Systems [https://github.com/microsoft/recommenders]
* Source of the data [https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-shop] (Thanks to REES46 Marketing Platform for this dataset.)
