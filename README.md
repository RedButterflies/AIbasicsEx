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

Task 4.1. Independent Application of PCA
Download the dataset 'voice_extracted_features.csv' from the course website. Perform PCA analysis on it:
1. Divide the dataset into subsets,
2. Generate a scatter plot leaving 2 principal components, use class labels to mark the colors of points corresponding to individual observations,
3. Build a plot of the explained variance percentage, select the optimal number of features for a 95% threshold,
4. Conduct classification by creating your own Pipeline object.

Task 4.2. Multiple Classification Experiments
Conduct 30 cycles of training and testing for the loaded dataset. Test the kNN, SVM, and Decision Tree algorithms. For each testing cycle, generate a confusion matrix. Average the matrices, consider which gender the methods detect more easily. Justify your answer. Hint: try to find justification in the confusion matrix.

Task 4.3. Definition of a Custom Class Determining the Optimal Number of Principal Components for PCA
Write a class whose object can be attached to a Pipeline object. Let this class allow the selection of the number of features based on the percentage of explained variance. Hint: to attach the class object to a Pipeline, it must have methods such as 'fit(x)', 'transform(x)', 'fit_transform(x)'.

Task 4.4. Definition of a Custom Class Responsible for Removing Outliers
Write a class whose object can be attached to a Pipeline object. This class is intended to find outliers and eliminate them by replacing them with the mean.


Task 5.2. Visualization of the Learning Process
Implement the training process of a neural network independently. Generate plots for metrics and loss functions for both the training and testing sets. Use the MNIST dataset as input data, the loading of which is demonstrated in Listing 5.8.

Listing 5.8. Loading the MNIST Dataset
from sklearn.datasets import load_digits
data = load_digits()
X = data.data
y = data.target


Task 5.3. Cross-Validation
Write a script that allows for grid search to find the most promising hyperparameter values. Test parameters such as the number of layers, the number of neurons in each layer, activation function, optimizer, and learning rate. Consider overfitting, meaning that the best result may not necessarily occur after the last epoch of network training.




