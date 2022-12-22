Random Forest Classifier in SciKitLearn with LIME for Binary Classification - Base problem category as per Ready Tensor specifications.

- random forest
- ensemble
- binary classification
- XAI
- interpretable
- LIME
- python
- pandas
- numpy
- sklearn
- scikit-optimize
- flask
- nginx
- uvicorn
- docker

This is an explainable version of Random Forest classifier, with LIME (Local interpretable Model-agnostic Explanations).

The classifier starts by creating an ensemble of decision trees and assigns the sample to the class that is predicted by the majority of the decision trees.

Model explainability is provided using LIME. Local explanations at each instance can be obtained.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

Hyperparameter Tuning (HPT) is conducted by finding the optimal number of decision trees to use in the forest, number of samples required to split an internal node, and number of samples required to be at a leaf node.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, interpret for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides three endpoints- /ping for health check, /infer for predictions in real time and /explain to generate local explanations.
