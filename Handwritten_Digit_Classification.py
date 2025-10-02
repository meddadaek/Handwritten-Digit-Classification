import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import catboost as cb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X, y = digits.data, digits.target

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("First 10 labels:", y[:10])

# Visualize one digit
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Label: {digits.target[0]}")
plt.show()
 
scale = MinMaxScaler()
X_scaled = scale.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "CatBoost": cb.CatBoostClassifier(verbose=0, random_state=42, iterations=600, learning_rate=0.1, depth=6)
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{name} Macro F1-score:", f1_score(y_test, y_pred, average='macro'))
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))

    
param = {
    "RandomForest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "DecisionTree": {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "CatBoost": {
        'iterations': [300, 600],
        'learning_rate': [0.01, 0.1],
        'depth': [4, 6]
    }
}

for name, model in models.items():
    grid = GridSearchCV(model, param[name], cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)
    print(f"Best Params for {name}:", grid.best_params_)
    print(f"Best CV Score for {name}:", grid.best_score_)
    
    best_model = grid.best_estimator_
    y_pred_tuned = best_model.predict(x_test)
    print(f"Tuned {name} Accuracy:", accuracy_score(y_test, y_pred_tuned))
    print(f"Tuned {name} Macro F1-score:", f1_score(y_test, y_pred_tuned, average='macro'))
    print(f"Tuned {name} Classification Report:\n", classification_report(y_test, y_pred_tuned))
    
    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Tuned {name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(digits.target_names))
    plt.xticks(tick_marks, digits.target_names, rotation=45)
    plt.yticks(tick_marks, digits.target_names)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
