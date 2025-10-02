# 🔢 Handwritten Digit Classification with ML

This project focuses on **classifying handwritten digits** (0–9) using different Machine Learning algorithms.  
It is based on the classic **Digits Dataset** provided by `scikit-learn`.  

The goal is to **train, evaluate, and tune models** such as:  
🌲 Random Forest, 🌳 Decision Tree, and 🐱 CatBoost — then compare their performance using accuracy, F1-scores, and confusion matrices.  

---

## ✨ What this Project Does
- 📥 Loads the **digits dataset** with `sklearn.datasets.load_digits`  
- 🖼️ Displays a sample digit image with its label  
- ⚖️ Scales the features using **MinMaxScaler**  
- 🧠 Trains and evaluates 3 ML models:  
  - 🌲 **Random Forest Classifier**  
  - 🌳 **Decision Tree Classifier**  
  - 🐱 **CatBoost Classifier**  
- 📊 Prints out:
  - Accuracy  
  - Macro F1-score  
  - Detailed classification report  
- 🔍 Performs **GridSearchCV** for hyperparameter tuning  
- 🎨 Plots confusion matrices for tuned models  

---

## 📦 Requirements

Before running, install the following Python libraries:  

pandas
numpy
matplotlib
scikit-learn
catboost

python
Copier le code

👉 Install all at once with:
```bash
pip install pandas numpy matplotlib scikit-learn catboost
▶️ How to Run
Save the script as main.py and run:

bash
Copier le code
python main.py
When executed, the program will:

🧾 Print dataset shape (X, y) and first labels

👁️ Show an example digit image

🤖 Train baseline models (RandomForest, DecisionTree, CatBoost)

📈 Print metrics (Accuracy + F1-score + Classification Report)

🔧 Use GridSearchCV to find best hyperparameters

🏆 Train tuned models and compare performance

🖼️ Plot confusion matrices for each tuned model

📊 Example Output
✅ Dataset Info
yaml
Copier le code
Shape of X: (1797, 64)
Shape of y: (1797,)
First 10 labels: [0 1 2 3 4 5 6 7 8 9]
🤖 Model Accuracy (example)
yaml
Copier le code
RandomForest Accuracy: 0.97
DecisionTree Accuracy: 0.84
CatBoost Accuracy: 0.98
🔧 Tuned Parameters (example)
rust
Copier le code
Best Params for RandomForest: {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2}
Best Params for DecisionTree: {'max_depth': 20, 'min_samples_split': 2}
Best Params for CatBoost: {'iterations': 600, 'learning_rate': 0.1, 'depth': 6}
📉 Confusion Matrix
The confusion matrix will be displayed as a heatmap showing how well each digit is classified.

📖 Explanation of Models
🌲 Random Forest: An ensemble method using multiple decision trees to improve accuracy and reduce overfitting.

🌳 Decision Tree: A simple model that splits data based on feature thresholds, good for interpretability but prone to overfitting.

🐱 CatBoost: A gradient boosting algorithm that handles categorical data well and often outperforms traditional models.

🔮 Future Improvements
🧠 Add Deep Learning models (CNNs with TensorFlow/PyTorch) for higher accuracy

💾 Save trained models with joblib for reuse

🌐 Create a small web app (Streamlit/Flask) to predict digits interactively

📱 Extend to mobile apps for handwriting recognition


