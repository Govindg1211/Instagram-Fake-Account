# Instagram-Fake-Account

---

## Introduction to Data Set

With the growth of social media, fake accounts have become a significant challenge,
affecting digital security and engagement authenticity. This dataset is designed to aid in
detecting fake Instagram accounts, comprising 576 training records and 120 test records, each
with 12 attributes describing an account’s characteristics.

Key profile details include username structure, bio length, follower count, number of posts,
and privacy settings. These features help identify patterns that differentiate real accounts (0)
from fake ones (1). The dataset is valuable for fraud detection, machine learning
classification, and social media security analysis.

It serves as a useful resource for researchers, data scientists, and cybersecurity professionals
developing machine learning models for fake account detection. This dataset supports feature
engineering, classification modeling, and fraud analysis to strengthen digital security and
authenticity. By studying these profiles, users can identify distinguishing traits of real and
fake accounts, contributing to improved trust in online interactions.

---

## Problem Statement: Predicting Fake Account on Instagram

With the increasing prominence of social media platforms like Instagram, the prevalence of
fake accounts has become a major issue. These fraudulent accounts are frequently used for
spreading misinformation, engagement fraud, cyber scams, and impersonation. Given the
vast number of users, manually distinguishing between real and fake accounts is highly
impractical. Therefore, an automated system leveraging machine learning is essential for
detecting such accounts efficiently and accurately

This project focuses on developing a machine learning model to predict whether an
Instagram account is real (0) or fake (1) based on various profile attributes. The dataset
includes essential features such as profile picture availability, username characteristics, bio
length, number of posts, follower and following counts, and privacy settings. By analyzing
these attributes, the model will be able to identify patterns and classify accounts accordingly.

By training and testing the model on these features, we aim to create a reliable solution
capable of analyzing account characteristics and detecting fraudulent activity. This can be
particularly beneficial for social media platforms, brands, and cybersecurity professionals
working to enhance digital trust and mitigate online threats. A robust classification model can
help in reducing the spread of fake profiles, ensuring safer user interactions, and improving
the overall authenticity of social media engagements.

The ultimate goal of this project is to build an efficient predictive model that can accurately
classify Instagram accounts as real or fake based on their attributes. This will contribute to
the development of advanced fraud detection mechanisms, strengthening the security and
credibility of social media platforms.

---

## Objective:

1. Data Collection & Preprocessing

❖ Load and examine the dataset to assess the distribution of real and fake accounts.

❖ Clean the data by handling missing values, outliers, and inconsistencies for better
reliability.


2. Feature Engineering

❖ Identify key attributes that play a significant role in detecting fake accounts.

❖ Apply feature scaling, transformation, and selection techniques to enhance model
performance.


3. Model Development

❖ Train and test various machine learning models, including Logistic Regression,
Decision Trees, Random Forest, SVM, and Neural Networks.

❖ Fine-tune hyperparameters to improve model accuracy and generalization.


4. Performance Evaluation

❖ Assess model effectiveness using metrics such as accuracy, precision, recall, F1-
score, and ROC-AUC.

❖ Compare different models and select the most accurate and efficient one.


5. Deployment & Real-World Application

❖ Develop a deployable system capable of analyzing Instagram accounts and
identifying potential fake profiles.

❖ Provide valuable insights to social media platforms, brands, and cybersecurity
professionals to enhance fraud detection and online security.

---

## Key Features to Consider for Fake Account Detection


1. Profile Information

• Profile Picture (profile pic): Indicates whether the account has a profile picture (1) or
not (0). Fake accounts often lack profile images.

• Username Structure (nums/length username): Measures the ratio of numerical
characters to the total length of the username. Fake accounts typically have usernames
with a high proportion of numbers.

• Full Name Characteristics (fullname words, nums/length fullname): Analyzes the
number of words in the full name and the proportion of numbers in it. Fake accounts
may display irregular name patterns.

• Username-Name Match (name==username): Verifies if the username and full name
are identical. Fake accounts often use the same name for both.


2. Bio & External Links

• Bio Length (description length): Represents the number of characters in the
account’s bio. Fake accounts might have either very short or excessively generic bios.

• External URL Presence (external URL): Indicates whether the account includes an
external link. Fake accounts often use URLs for promotional or malicious purposes.

3. Account Privacy & Activity

• Privacy Status (private): Identifies whether the account is private (1) or public (0).
Fake accounts can be either, but public accounts are more likely to engage in spam
activities.

• Number of Posts (#posts): The total number of posts made by the account. Fake
accounts typically have very few or no posts.

4. Social Engagement Metrics

• Number of Followers (#followers): The number of followers the account has. Fake
accounts may have unusually high or low follower counts.

• Number of Accounts Followed (#follows): The number of accounts the user follows.
Fake accounts often follow many profiles but have few followers in return.

5. Target Variable

• Fake Account Label (fake): The dependent variable indicating whether the account is
real (0) or fake (1)

---

## Techniques Used in the Notebook

1. Data Preprocessing

• Libraries such as pandas, numpy, matplotlib, and seaborn are utilized for data
manipulation and visualization tasks.

• Handle missing data by using .isnull().sum() and calculating the percentage of missing
values.

• Check dataset details using .info(), .describe(), and .shape() to understand its structure
and summary statistics.


2. Exploratory Data Analysis (EDA)

• View the first and last few rows of the dataset using .head() and .tail().

• Generate a statistical summary of the dataset using .describe(), covering both
numerical and categorical data.

• Identify unique values in categorical columns to understand the data distribution.


3. Feature Scaling

• Standardize the dataset using the StandardScaler from sklearn.preprocessing, which
scales features to have a mean of zero and a standard deviation of one.

• This step is crucial for models like Logistic Regression and SVM, as they are sensitive
to the scale of features.


4. Feature Selection (RFE & Mutual Information)

• Recursive Feature Elimination (RFE): A technique for selecting important features by
repeatedly fitting the model and eliminating the least significant features.

• Mutual Information Classification: Assesses the relationship between each feature and
the target variable to identify and select the most relevant features.

---

## Machine Learning Model Implementation

1. Model Selection - Supervised Learning Algorithms

• Random Forest Classifier: An ensemble learning technique that combines multiple
decision trees to enhance classification accuracy.

• Logistic Regression: A statistical method used for binary classification problems.

• Support Vector Machine (SVM): A classification algorithm that identifies the optimal
hyperplane to separate different classes.


2. Model Training and Predictions

• Each model is trained using model.fit(X, y), where X represents the training data and y
is the target variable.

• Predictions are made for both training and testing datasets using predict().

• If the model supports probability estimation with predict_proba(), it is used to compute
the ROC-AUC score.


3. Cross-Validation for Model Performance

• Stratified K-Fold Cross-Validation: Uses StratifiedKFold with 5 splits to ensure
balanced class distribution in each fold.

• cross_val_score computes accuracy scores across multiple folds to evaluate the
model’s ability to generalize.


4. Model Evaluation Metrics
Different evaluation metrics are used to measure model performance:

• Accuracy Score: Evaluates the overall correctness of the model.

• Precision Score: Calculates the proportion of true positives out of all predicted
positives.

• Recall Score: Assesses how effectively the model identifies actual positives.

• F1 Score: The harmonic mean of precision and recall, providing a balance between the
two.

• ROC-AUC Score: Measures the model’s ability to rank positive instances higher than
negative ones.

---

## Steps for Project Implementation

Step 1: Import Required Libraries

• Import the necessary Python libraries for data manipulation, visualization, and
machine learning tasks.

• Load essential tools for data preprocessing, model building, and evaluation.


Step 2: Load the Dataset

• Load the dataset and examine its structure to understand its contents.


Step 3: Data Preprocessing

• Address missing values, perform feature engineering, and remove any irrelevant
columns to prepare the data.


Step 4: Define Features & Targets

• Separate the dataset into input features (X) and target variables (y) for the model.


Step 5: Feature Scaling

• Apply feature scaling using StandardScaler to standardize the data and enhance model
performance.


Step 6: Train-Test Split

• Divide the dataset into training and testing sets to evaluate model performance.


Step 7: Define Machine Learning Models

• Implement a Stacking Regressor using multiple base models for improved predictions.


Step 8: Train the Models

• Train the Stacking Regressor using the training data to fit the model.


Step 9: Model Predictions

• Use the trained model to predict the target variable values for both the training and
testing datasets.


Step 10: Model Evaluation

• Evaluate model performance using metrics such as MAE (Mean Absolute Error), MSE
(Mean Squared Error), and R² Score.

---

## Summary

This project aimed to develop a machine learning model to predict the number of subscribers
and course ratings for Udemy courses by leveraging advanced data analysis and modeling
techniques. The dataset underwent comprehensive preprocessing to ensure data quality and
enhance predictive accuracy. This included handling missing values, performing feature
engineering by creating meaningful attributes such as the ratio of subscribers per review and
lecture density, and applying feature scaling using StandardScaler to standardize numerical
features. Additionally, to optimize computational efficiency and improve model performance,
Principal Component Analysis (PCA) was employed to reduce dimensionality while
preserving significant variance in the dataset.

For predictive modeling, a Stacking Regressor was implemented to integrate multiple
machine learning algorithms, allowing for a more robust and accurate prediction framework.
The base models included Random Forest, XGBoost, Decision Tree, Extra Trees, and
CatBoost, which were selected for their strong performance in regression tasks. To further
enhance predictive capabilities, a Neural Network (Multi-Layer Perceptron Regressor) was
used as the meta-model, leveraging deep learning techniques to refine final predictions. The
models were trained and rigorously evaluated using key performance metrics, including
Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score, ensuring that the
predictions were both reliable and interpretable. Through this approach, the project
successfully provided insights into the factors influencing Udemy course popularity and
ratings, demonstrating the effectiveness of machine learning in analyzing educational
platforms.

---

## Conclusion
The Stacking Regressor approach proved to be a highly effective method for predicting both
course popularity (subscriber count) and quality (ratings) by integrating multiple machine
learning models. This ensemble technique enhanced accuracy and robustness compared to
individual models, demonstrating the advantages of leveraging diverse algorithms for better
generalization. The results highlight the potential of machine learning in optimizing content
strategies, allowing course creators, marketers, and e-learning platforms to make data-driven
decisions regarding pricing, promotion, and course design.

By understanding the key factors influencing course success, stakeholders can refine their
offerings to better align with learner preferences and industry trends. Future enhancements to
the model could include advanced hyperparameter tuning, additional feature engineering, and
deep learning techniques for improved predictive performance. Incorporating time-series
analysis, sentiment analysis on course descriptions, or engagement metrics could further
enhance the model’s ability to identify emerging trends in online education, making it a
valuable tool for e-learning platforms.

---
















