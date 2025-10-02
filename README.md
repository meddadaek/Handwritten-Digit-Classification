# Handwritten-Digit-Classification



This project trains and evaluates multiple machine learning models on the classic **digits dataset** (`sklearn.datasets.load_digits`). It compares **Random Forest**, **Decision Tree**, and **CatBoost** classifiers, applies hyperparameter tuning, and visualizes confusion matrices.

---

## 📌 Features
- Loads and preprocesses the **digits dataset**.
- Applies **MinMax scaling** for normalization.
- Trains three models:
  - **RandomForestClassifier**
  - **DecisionTreeClassifier**
  - **CatBoostClassifier**
- Evaluates models using:
  - Accuracy
  - F1-score (Macro average)
  - Classification reports
- Performs **GridSearchCV** for hyperparameter tuning.
- Visualizes results with **confusion matrices**.

Install dependencies:

bash
Copier le code
python main.py
The script will:

Print dataset info (shape and labels).

Show an example digit.

Train baseline models and print evaluation results.

Perform hyperparameter tuning with GridSearchCV.

Print best hyperparameters and tuned evaluation metrics.

Display confusion matrices for each tuned model.

📊 Example Output
Accuracy and F1-scores for each model.

Best hyperparameters from GridSearchCV.

Confusion matrices visualizing model predictions.


📦 requirements.txt
nginx
Copier le code
pandas
numpy
matplotlib
scikit-learn
catboost
pgsql
Copier le code
