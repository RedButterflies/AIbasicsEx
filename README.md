# AIbasicsEx
Exercises from my AI basics class
Translation of the task in English:
Task 2.1. First Glance
Download the file "housing.xlsx" from the course website. Similar to tasks from previous classes (Task 1.4), generate a correlation matrix for the loaded dataset. Analyze the correlation matrix. What dependencies may be meaningful, and which ones are likely random? Generate plots of the correlation between independent features and the dependent feature (median housing price).

Task 2.2. Multiple Testing of the Model
Modify the script that shows Listing 2.4 to create a script that allows for multiple tests of a linear regression model. Place the script in a separate function that takes the number of repetitions as an argument. Hint: use a for loop in the function. Each time, the dataset should be split into training and testing subsets randomly; do not provide the random_state argument to ensure different results each time. The result of the experiment should be the average value of the mean_absolute_percentage_error metric, meaning the average percentage error of the regression.

Task 2.3. Handling Outliers
Perform Task 2.2, adding a procedure to remove/replace outlier values. Compare the results obtained in the previous task with the new results.

Task 2.4. Generation of New Features
Attempt to propose features/feature combinations that could improve the quality of linear regression predictions. Test the proposed solutions.

Task 2.5. Independent Data Analysis
Using the code presented in Listing 2.9, load the Diabetes dataset. Analyze it similarly to how we did with the "Boston Housing" dataset.

Listing 2.9. Code to load the diabetes dataset:
from sklearn.datasets import load_diabetes
data = load_diabetes()




Task 3.2. Transformation of Binary Qualitative Features

Create a function based on the script that presents Listing 3.1 to a function with the following header:

def qualitative_to_0_1(data, column, value_to_be_1):

The function is intended to return the variable 'data' with the column named 'column' containing values of 0 or 1. Assign the value 1 to the values of the 'column' equal to 'value_to_be_1'. Use the created function to transform the rest of the binary qualitative features.

Task 3.3. Calculation of Classification Metric Values

For the data presented in Table 3.3 and Table 3.4, determine the values of all classification metrics.

Task 3.4. Impact of Parameters on the Quality of Classification Models

Test the kNN and SVM algorithms with different parameters:
- kNN: number of neighbors (n_neighbors), method of determining weights ('uniform', 'distance'),
- SVM: kernel.

Task 3.5. Impact of Scaling Methods

Examine the influence of different scaling methods on the classification quality using SVM and kNN methods. Test the classes StandardScaler, MinMaxScaler, RobustScaler.



Task 3.6. Independent Classification

Using the code presented in Listing 3.7, load data to train a model to distinguish between benign and malignant tumors. Conduct an analysis using the discussed classification methods. Build a decision tree with a height of 5, generate its illustration, and discuss which features influence the outcome.

Listing 3.7. Script allowing loading of the BreastCancer dataset

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
