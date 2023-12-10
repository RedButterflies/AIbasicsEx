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
