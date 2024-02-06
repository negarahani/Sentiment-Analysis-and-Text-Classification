# sentiment-analysis-and-text-classification


This project contains a Jupyter Notebook created in Google Colab for sentiment analysis using various machine learning algorithms.

## Description

The Jupyter Notebook implements sentiment analysis on a dataset containing customer reviews and star ratings. The notebook includes data preprocessing steps, feature extraction using Bag of Words (BoW) and TF-IDF methods, and training/testing of machine learning models such as Perceptron, Support Vector Machine (SVM), Logistic Regression, and Naive Bayes.

## Package Requirements

To run the code, make sure the following packages are installed:

- `pandas=1.5.3=py311heda8569_0`
- `numpy=1.24.3=py311hdab7c0b_1`
- `nltk=3.8.1=py311haa95532_0`
- `scikit-learn=1.3.0=py311hf62ec03_0`
- `scikit-learn-intelex=2023.1.1=py311haa95532_0`
- `contractions`

The Anaconda Python distribution is recommended as it typically comes with these packages pre-installed.

## Usage

1. Download the dataset from [this link](https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz).
2. Save the downloaded file as `data.tsv` in the same folder as the Jupyter Notebook.
3. Install the required packages listed in the package requirements section.
4. Execute the code cells in the notebook to reproduce the sentiment analysis results.


## Overview

The notebook is divided into several sections:

1. **Data Preparation:** Reading and preprocessing the dataset.
2. **Feature Extraction:** Extracting features using Bag of Words (BoW) and TF-IDF methods.
3. **Model Training:** Training machine learning models including Perceptron, SVM, Logistic Regression, and Naive Bayes.
4. **Evaluation:** Evaluating the performance of the trained models using metrics like accuracy, precision, recall, and F1-score.


