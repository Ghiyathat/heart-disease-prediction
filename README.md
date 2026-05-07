# heart-disease-prediction
## Table of Contents
[1. Project Overview](1-project-overview)

[2. The Dataset](2-the-dataset)

[3. The Analytics Stack](3-the-analytics-stack)

[4. Data Cleaning & Feature Engineering](4-data-cleaning-&-feature-engineering)

[5. Exploratory Data Analysis (EDA)](5-exploratory-data-analysis-(EDA))

[6. Modeling & Prediction Strategy](6-modeling-&-prediction-strategy)

[7. Key Outcomes & Model Performance](7-key-outcomes-&-model-performance)

[8. Challenges & Limitations](8-challenges-&-limitations)

### **1. Project Overview**

This project focuses on building a predictive framework to identify individuals at high risk of heart disease using the **2020 CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset. With cardiovascular diseases being a leading cause of mortality globally, the goal was to leverage machine learning to move beyond traditional clinical observation and toward data-driven early detection.

Using a dataset of over **300,000 survey respondents**, I explored the relationship between 18 health indicators—ranging from lifestyle choices like smoking and alcohol consumption to clinical markers like BMI and pre-existing conditions like diabetes. 

The core of this task involved:
* **Predictive Modeling:** Implementing and comparing the **K-Nearest Neighbors (KNN)** and **Decision Tree** algorithms to classify patient risk.
* **Risk Factor Analysis:** Identifying which habits and health conditions have the strongest statistical correlation with heart disease.
* **Clinical Utility:** Evaluating models not just on overall accuracy, but on **Recall** and **AUC (Area Under the Curve)**, ensuring the system is sensitive enough to catch true positive cases in a healthcare setting.

### **2. The Dataset**

* **Total Observations:** 319,795 records.
* **Target Variable:** `HeartDisease` (Binary: Yes/No)—indicating whether the respondent has ever been diagnosed with coronary heart disease or myocardial infarction.
* **Feature Diversity:** The dataset contains **18 variables** categorized into three main types:
    * **Lifestyle Factors:** Smoking status, alcohol consumption, physical activity, and average sleep time.
    * **Clinical Indicators:** BMI (Body Mass Index), physical health (days ill/injured), and mental health.
    * **Demographics & Comorbidities:** Age category, race, sex, and pre-existing conditions like Diabetes, Asthma, Kidney Disease, and Skin Cancer.

### **3. The Analytics Stack**

* **Python:** The primary programming language used for the entire end-to-end pipeline.
* **Pandas & NumPy:** These were the workhorses for data manipulation. They allowed me to handle the 300,000+ rows of data, perform column transformations, and execute mathematical operations efficiently.
* **Scikit-Learn (Sklearn):** The core machine learning library. I used it for:
    * **Preprocessing:** `StandardScaler` for normalizing numerical data and `OneHotEncoder` for converting text-based categories into machine-readable formats.
    * **Algorithms:** Implementing the **K-Nearest Neighbors (KNN)** and **Decision Tree** classifiers.
    * **Evaluation:** Using `metrics` to calculate accuracy, precision, and the ROC-AUC score.
* **Matplotlib & Seaborn:** These visualization libraries were used to translate complex data into clear charts. I specifically used them to plot the **ROC Curves** and the **Evaluation Metric comparisons**, making it easier to see which model performed better at a glance.
* **Jupyter Notebook:** The interactive environment used to document the code, visualize outputs, and maintain a clear, step-by-step analytical narrative.

### **4. Data Cleaning & Feature Engineering**

Raw medical data is rarely ready for a machine learning model. To ensure the algorithms could process the survey results accurately, I implemented a preprocessing pipeline focused on converting text to numbers and normalizing scales.

* **Handling Categorical Variables:** Machine learning models require numerical input, but much of our data (like `Sex`, `AgeCategory`, and `Race`) was text-based. I used **One-Hot Encoding** via Scikit-Learn’s `make_column_transformer` to transform these categories into binary (0 or 1) columns. This prevents the model from assuming a mathematical "order" where none exists (e.g., assuming "White" is "greater than" "Black").
* **Feature Scaling:** Features like `BMI` range from 12 to 90+, while `SleepTime` ranges only from 1 to 24. To prevent the model from being biased toward features with larger raw numbers, I applied **StandardScaler**. This centered the data around zero, ensuring every feature contributed equally to the final prediction.
* **Target Encoding:** The primary target, `HeartDisease`, was converted from "Yes/No" strings into a simple **1/0 binary format** to allow for mathematical classification.
* **Data Splitting:** I utilized an **80/20 Train-Test Split**. This means the model learned from 80% of the data, while the remaining 20% was kept completely separate to act as an "unseen exam," proving whether the model could actually predict heart disease in new patients.
* **Column Selection:** I carefully selected 18 features that had high clinical relevance, ensuring the model wasn't distracted by administrative data, but instead focused on the intersection of lifestyle and biology.

```
# Importing libraries
import pandas as pd
import numpy as np

%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.compose import make_co
--------------------
# Read dataset 
df = pd.read_csv('heart_2020_cleaned.csv')
df.head()
--------------------
 df.info()
--------------------
df.nunique()
--------------------
df = df.replace({'Yes':1, 'No':0, 'Male':1, 'Female':0, 'No, borderline diabetes':0, 'Yes (during pregnancy)':1})
df = df.infer_objects(copy=False)
--------------------
plt.figure(figsize=(10, 6))
# Using a countplot is more appropriate for categorical data like Sex
sns.countplot(data=df, x='Sex', hue='HeartDisease', palette=['#fccc79', 'red'])
plt.title('Distribution of Heart Disease by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.legend(
--------------------
# Set the figure size
plt.figure(figsize=(13, 6))

# Use countplot for categorical data
# hue='HeartDisease' automatically splits the bars by the outcome
sns.countplot(data=df, x='Smoking', hue='HeartDisease', palette=['#fccc79', 'red'])

# Add labels and title
plt.xlabel("Smoking Status (0 = Non-Sm
--------------------
plt.figure(figsize=(13, 6))

# Refined countplot call
sns.countplot(data=df, x='Race', hue='HeartDisease', palette='YlOrBr')

plt.title('Frequency of Heart Disease by Race')
plt.xlabel('Race')
plt.ylabel('Frequency')

# Adding a title to the legend for better clarity
plt.legend(title='Heart Disease'
--------------------
plt.figure(figsize=(13, 6))

# Refined countplot: passing column names directly as strings to 'x' and 'hue'
sns.countplot(data=df, x='AgeCategory', hue='HeartDisease', palette='YlOrBr')

# Using plt.title for a more robust notebook execution
plt.title("Distribution of Heart Disease Cases by Age Cate
--------------------
# Set the figure size
plt.figure(figsize=(13, 6))

# Use countplot for categorical data comparison
sns.countplot(data=df, x='KidneyDisease', hue='HeartDisease', palette=['#fccc79', 'red'])

# Add labels and title
plt.xlabel("Kidney Disease (0 = No, 1 = Yes)")
plt.ylabel("Frequency")
plt.title("Distr
--------------------
# Set the figure size
plt.figure(figsize=(13, 6))

# Use countplot for categorical comparison
# data=df refers to the heart disease dataframe
sns.countplot(data=df, x='SkinCancer', hue='HeartDisease', palette=['#fccc79', 'red'])

# Add labels and title
plt.xlabel("Skin Cancer History (0 = No, 1 = Ye
--------------------
# Set the figure size
plt.figure(figsize=(13, 6))

# Use countplot for categorical comparison between Stroke and HeartDisease
# This clearly shows the frequency without the gaps created by histogram bins
sns.countplot(data=df, x='Stroke', hue='HeartDisease', palette=['#fccc79', 'red'])

# Add labels
--------------------
# Set the figure size
plt.figure(figsize=(13, 6))

# Use countplot for categorical comparison between Diabetic and HeartDisease
# This clearly groups the categories (Yes, No, Borderline, etc.)
sns.countplot(data=df, x='Diabetic', hue='HeartDisease', palette=['#fccc79', 'red'])

# Add labels and titl
--------------------
# Ensuring to only calculate correlation for numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr().round(2)

plt.figure(figsize=(14, 7))

# Plotting the heatmap with annotations and a clear color map
sns.heatmap(correlation, annot=True, cmap='YlOrBr', fmt=
--------------------
sns.set_style('white')
plt.figure(figsize=(13, 6))

# Calculating absolute correlation with the target variable
# Sort and exclude the target itself (the last one)
abs_corr = abs(correlation['HeartDisease']).sort_values()[:-1]

# Plotting with the YlOrBr palette
abs_corr.plot.barh(color=sns.color_pa
--------------------
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(13, 5))

# Plotting KDE for HeartDisease cases vs. Normal cases
# Changed 'shade' to 'fill' to avoid deprecation warnings
sns.kdeplot(data=df[df["HeartDisease"]==1], x="BMI", alpha=0.5, fill=True, color="red", label="HeartDisease", ax=ax)

--------------------
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(13, 5))

# Plotting KDE for HeartDisease cases vs. Normal cases
# Updated 'shade' to 'fill' for future compatibility
sns.kdeplot(data=df[df["HeartDisease"]==1], x="SleepTime", alpha=0.5, fill=True, color="red", label="HeartDisease", ax=ax)
--------------------
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(13, 5))

# Plotting KDE for HeartDisease cases vs. Normal cases
# Updated 'shade' to 'fill' for compatibility
sns.kdeplot(data=df[df["HeartDisease"]==1], x="PhysicalHealth", alpha=0.5, fill=True, color="red", label="HeartDisease", ax=ax)
s
--------------------
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(13, 5))

# Plotting KDE for HeartDisease cases vs. Normal cases
# fill=True is the modern replacement for shade=True
sns.kdeplot(data=df[df["HeartDisease"]==1], x="MentalHealth", alpha=0.5, fill=True, color="red", label="HeartDisease", ax=
--------------------
#Split dataset for training and testing
#Select Features
features = df.drop(columns =['HeartDisease'], axis = 1)

#Select Target 
target = df['HeartDisease']

# Set Training and Testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feat
--------------------
# Encoding
transformer = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ['AgeCategory', 'Race', 'GenHealth']),  # Changed sparse=False to sparse_output=False
    remainder='passthrough')

# Encode training data 
transformed_train = transformer.fit_transform(X_train)
transformed_tr
--------------------
# Standardization
scaler = StandardScaler()

# Scale trainint data
X_train = scaler.fit_transform(X_train)

# Scale test data
X_test = scaler.fit_transform(X_test)
--------------------
def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, 
--------------------
# building model
X_train
--------------------
# Building a neighbor through kneighbor classifier 
# Building a model using KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)


# Evaluate Model
knn_eval = evaluate_model(knn, X_test, y_test)

# Print result
print('Accuracy:', knn_eval['acc'])
print('Preci
--------------------
# Building Decision Tree model 
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Evaluate Model
clf_eval = evaluate_model(clf, X_test, y_test)

# Print result
print('Accuracy:', clf_eval['acc'])
print('Precision:', clf_eval['prec'])
print('Recall:', clf_eval['rec'])
pri
--------------------
# Initialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Model Comparison: Decision Tree vs. KNN', fontsize=16, fontweight='bold')
fig.set_facecolor('white')

# --- First Plot: Evaluation Metrics ---
barWidth = 0.3
clf_score = [clf_eval['acc'], clf_eval[
--------------------

```
### **5. Exploratory Data Analysis (EDA)**

In this stage, I identified which factors most strongly distinguished healthy individuals from those with heart disease by visualizing correlations and distributions,

* **Demographic Vulnerabilities:** I analyzed the distribution of heart disease across **Age Categories** and **Sex**. The data showed a clear trend: heart disease cases were significantly more frequent in older age groups (65+) and among male respondents, confirming age and sex as primary non-modifiable risk factors.
* **The Lifestyle Link:** Using Seaborn countplots, I compared **Smoking** and **Alcohol Drinking** status. The analysis visually demonstrated that smokers had a noticeably higher frequency of heart disease compared to non-smokers, providing a clear statistical link between lifestyle choices and cardiovascular health.
* **Comorbidity Correlation:** I explored the overlap between heart disease and other chronic conditions like **Kidney Disease**, **Stroke**, and **Diabetes**. The findings showed that individuals with a history of stroke or kidney disease had a disproportionately higher likelihood of also having heart disease.
* **Physical & Mental Markers:** I used **KDE (Kernel Density Estimate) plots** to compare the distributions of **BMI**, **SleepTime**, and **Physical Health**. 
    * **BMI:** Patients with heart disease showed a distribution shifted toward higher BMI values (obesity range).
    * **Physical Health:** There was a sharp spike in heart disease cases among people who reported more than 20 days of "poor physical health" in a month.
* **Statistical Correlation Map:** I generated a **Correlation Heatmap** to quantify the relationships between all numerical variables. This heatmap identified `PhysicalHealth` and `DiffWalking` (difficulty walking) as having some of the strongest positive correlations with the target variable, `HeartDisease`.

These insights helped validate that the features selected were indeed relevant for the modeling phase that followed.

### **6. Modeling & Prediction Strategy**

For the predictive phase of the project, I implemented a comparative modeling strategy. The goal was to test two distinct types of supervised learning algorithms to determine which provided the most reliable risk assessment for heart disease.

* **Model Selection:**
    * **K-Nearest Neighbors (KNN):** Chosen for its ability to find patterns based on "similarity." If a patient’s health markers are similar to others who have heart disease, the model groups them together. I initialized this with $k=5$ neighbors.
    * **Decision Tree Classifier:** Chosen for its transparency. Decision trees mimic medical diagnostic flowcharts, making it easy to see which specific health triggers (like "Age > 65" or "Smoker = Yes") lead to a high-risk classification.
* **The Evaluation Framework:**
    In healthcare, **Accuracy** alone is a dangerous metric because the dataset is imbalanced. If a model simply predicts "No" for everyone, it would be 91% accurate but would fail to save a single life. Therefore, my strategy focused on:
    * **Recall (Sensitivity):** Ensuring we minimize "False Negatives"—we don't want to tell a sick person they are healthy.
    * **Precision:** Ensuring that when the model flags a patient, there is a high probability they truly need attention.
    * **ROC-AUC Score:** This served as the primary benchmark, measuring the model's ability to distinguish between the two classes across all possible thresholds.
* **Experimental Design:**
    I utilized a custom `evaluate_model` function to ensure both models were judged on a level playing field. Both models were trained on the same **80% training set** and validated on the **20% testing set**, ensuring that the results reflect real-world predictive power rather than just "memorizing" the training data.

### **7. Key Outcomes & Model Performance**

The modeling phase provided a direct comparison between a distance-based algorithm (KNN) and a rule-based algorithm (Decision Tree). By evaluating them on the 20% "unseen" test set, the following empirical results were achieved:

#### **Model Comparison Results:**

| Metric | K-Nearest Neighbors (KNN) | Decision Tree |
| :--- | :--- | :--- |
| **Accuracy** | **90.4%** | 86.4% |
| **Precision** | **37.6%** | 23.8% |
| **Recall (Sensitivity)** | 14.1% | **25.2%** |
| **F1-Score** | 20.5% | **24.5%** |
| **ROC-AUC Score** | **0.714** | 0.588 |

#### **Key Performance Insights:**
* **The Accuracy Trap:** While KNN achieved a high accuracy of **90.4%**, this is largely because it is very good at identifying the majority class (people without heart disease). In medical diagnostics, accuracy can be misleading due to the high volume of healthy patients in the dataset.
* **The Sensitivity Trade-off:** The **Decision Tree** model outperformed KNN in **Recall (25.2%)**. This means the Decision Tree was nearly twice as effective at actually "catching" people who had heart disease, whereas KNN was more conservative and missed more positive cases.
* **ROC-AUC Analysis:** KNN achieved a significantly higher **AUC of 0.714** compared to the Decision Tree's 0.588. This indicates that KNN has a better overall mathematical ability to distinguish between a healthy and a diseased heart across different risk thresholds, even if its default predictions were conservative.
* **Clinical Relevance:** The low precision across both models (under 40%) suggests that many patients flagged as "at risk" might not actually have the disease. However, in a screening context, a higher **Recall** (as seen in the Decision Tree) is often preferred to ensure no high-risk patient is overlooked, even at the cost of some false alarms.

### **8. Challenges & Limitations**

* **Significant Class Imbalance:** The most prominent challenge was the disparity in the target variable. Since only about **$8.6\%$** of patients had heart disease, the models were naturally biased toward predicting "No." This explains why there is high accuracy ($90\%+$) but lower **Precision and Recall**. Standard accuracy is not a reliable metric here; we had to rely on the **ROC-AUC** score to gauge true performance.
* **The Subjectivity of Survey Data:** Because the dataset is based on the **CDC’s BRFSS survey**, the features are self-reported. Variables like `GenHealth` (General Health) or `MentalHealth` are subjective and can vary significantly between individuals, adding "noise" to the model's predictive patterns.
* **Computational Intensity of KNN:** The **K-Nearest Neighbors** algorithm requires calculating the distance between points. With a dataset of over **$300,000$ records**, this became computationally expensive and slow during the training and evaluation phases compared to the Decision Tree.
* **Static Snapshot:** The data represents a snapshot from **2020**. While it provides a massive sample size, it does not account for changes in health trends or medical advancements that may have occurred since the survey was taken.
* **Hyperparameter Tuning:** In this iteration, I used standard configurations (like $k=5$ for KNN). There is significant potential to improve these results further by using techniques like **Grid Search** or **Random Search** to find the optimal settings for each model.
* **Lack of Direct Clinical Markers:** While BMI and age are useful, the model lacks direct clinical data such as blood pressure readings, cholesterol levels, or genetic markers, which are critical for a "gold-standard" medical diagnosis.

---

[↑ Back to Top](#table-of-contents)

