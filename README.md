**Streamlit app link**:https://boosting-jqprazyujccvmk2tft6nnv.streamlit.app/
# Gradient Boosting Regressor — Car Price Prediction

This project demonstrates how to use Gradient Boosting Regression to predict car selling prices based on various features. It includes data preprocessing, model training, hyperparameter tuning, cross-validation, and performance evaluation.

**Project Overview**

The goal of this project is to accurately predict car selling prices using machine learning.
The Gradient Boosting Regressor algorithm from scikit-learn is used because it builds strong predictive models by combining multiple weak learners (decision trees).

**The process includes:**

Handling categorical data using one-hot encoding

Applying a log transformation on the target variable to reduce skewness

Splitting the data into training and testing sets

Performing hyperparameter tuning with GridSearchCV

Evaluating model performance using RMSE and R² metrics

**Dataset**

The dataset contains car-related information such as selling price, year, kilometers driven, fuel type, transmission, and ownership details.
The target variable is selling_price, which is log-transformed during preprocessing to make the data more normally distributed.

**Workflow Explanation**

Importing Libraries
The required Python libraries are imported for data manipulation, model creation, parameter tuning, and evaluation.

Data Preprocessing
All categorical variables are converted into numeric form using one-hot encoding.
The selling price is log-transformed to handle skewed distribution and outliers.

Feature Selection and Splitting
The dataset is divided into input features (X) and the target variable (y).
Data is split into training (80%) and testing (20%) sets to evaluate model generalization.

Model Definition and Parameter Grid
The Gradient Boosting Regressor is initialized, and a range of hyperparameters (number of estimators, learning rate, and tree depth) is defined for tuning.

Cross-Validation and Grid Search
A five-fold cross-validation is performed to ensure stable and reliable model performance.
GridSearchCV is used to automatically find the best combination of hyperparameters.

Model Training and Prediction
The model is trained on the training data and used to predict selling prices for the test set.

Reversing Log Transformation and Evaluation
Predicted and actual values are converted back from the logarithmic scale.
Performance is measured using Root Mean Squared Error (RMSE) and R² Score to assess prediction accuracy.

Results Display
The best parameters found by GridSearchCV, RMSE, and R² Score are printed as the final output.

**Expected Output**

The model prints:

The best set of hyperparameters

The RMSE value (how far predictions are from actual prices)

The R² Score (how well the model explains the variance in selling prices)

A high R² and low RMSE indicate a strong predictive model.

**Requirements**

To run this project, the following Python libraries must be installed:

pandas

numpy

scikit-learn

**Key Insights**

Gradient Boosting works by sequentially building trees that correct the errors of previous trees.

Log transformation stabilizes variance and improves prediction accuracy.

Cross-validation provides more reliable performance estimates.

Hyperparameter tuning improves model accuracy and reduces overfitting.

Future Enhancements

Extend parameter tuning to include more variables such as subsampling rate and feature fraction.

Compare Gradient Boosting with advanced boosting models like XGBoost, LightGBM, or CatBoost.

Visualize feature importance to understand which attributes most affect selling price.
