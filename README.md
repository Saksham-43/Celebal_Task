# Machine Learning Model Comparison using GridSearchCV
 
**📌 Overview:**

This project showcases a complete machine learning workflow applied to a real-world dataset. The process includes data preprocessing, train-test splitting, and the application of various supervised learning algorithms. The primary goal is to compare the performance of different models using GridSearchCV for hyperparameter tuning to identify the best-performing model based on accuracy.


**✅ Models Used:**

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)



**🛠️ Key Steps:**

* **Data Preprocessing:** Handled missing values, filtered invalid entries, and transformed features to make the data ML-ready.

* **Train-Test Split:** Dataset split into training and testing sets to evaluate model generalization.

* **Model Implementation:** Applied multiple classification algorithms using scikit-learn.

* **Hyperparameter Tuning:** Used GridSearchCV to find the best parameters for each model.

* **Model Evaluation:** Compared model performance based on accuracy and cross-validation scores.



**📊 Results:**

Each model's accuracy and optimal hyperparameters (as found by GridSearchCV) are summarized and compared to determine the most suitable algorithm for the dataset.

Accuracy of Logistic Regression:  0.7947761194029851<br>
Decision Tree Accuracy: 0.7686567164179104<br>
Random Forest Accuracy: 0.8022388059701493<br>
K-Nearest Neighbors Accuracy: 0.6902985074626866<br>
Naive Bayes Accuracy: 0.7910447761194029<br>
SVM Accuracy: 0.6791044776119403<br>

**Accuracy After GridSearchCV:**

Accuracy of Logistic Regression: 0.7910447761194029<br>
Accuracy of Decision Tree: 0.7985074626865671<br>
Accuracy of Random Forest: 0.8022388059701493<br>
Accuracy of Naive Bayes: 0.7910447761194029<br>
Accuracy of SVM : 0.7686567164179104<br>
Best score for KNN: 0.7304<br>


**📁 Technologies Used:**

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib / Seaborn (if visualizations included)


**🚀 How to Run:**

1)Clone the repository<br>
2)Install required libraries<br>
3)Run the Jupyter Notebook or Python script provided<br>
