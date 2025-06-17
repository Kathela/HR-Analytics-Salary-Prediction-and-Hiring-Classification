# HR-Analytics-Salary-Prediction-and-Hiring-Classification
This project applies machine learning techniques to analyze and predict outcomes in a recruitment process. It is divided into two parts: Salary Prediction (Regression) and Hiring Classification (Classification)

This project focuses on analyzing human resources data to build predictive models for salary estimation and hiring decisions. It is divided into two main parts:

In Part 1, a regression analysis was conducted to predict the Salary of candidates based on features such as Experience, Education, and Position. After transforming categorical variables into dummy variables and standardizing the data, several linear models were applied, including Ordinary Least Squares (OLS), Ridge, Lasso, and ElasticNet regression. The performance of each model was evaluated using metrics such as RMSE and R² score, with Ridge regression slightly outperforming the others in terms of generalization.

In Part 2, the goal was to classify whether a candidate was hired (Contratado) based on the available features. A Logistic Regression model was first implemented, followed by a more powerful Random Forest Classifier. Both models were evaluated using a confusion matrix, ROC curve, AUC score, and a detailed classification report (precision, recall, and F1-score). Visualizations and clear performance metrics were used to assess the models' ability to correctly identify hired versus non-hired candidates.

This end-to-end project demonstrates practical applications of both regression and classification techniques using scikit-learn, and includes data preprocessing, model training, evaluation, and visualization.
