import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import time


# from sklearn.metrics import classification_report, confusion_matrix

def random_forest():
    print("Starting Random Forest")

    df = pd.read_csv("data/student_depression_dataset.csv", encoding="latin-1")

    # encode categorical variables
    df = pd.get_dummies(df)

    # drop rows with missing values
    df = df.dropna()

    X = df.drop(columns=['Depression'])
    y = df['Depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    
    # generate initial baseline Random Forest classifier and calculate runtime
    start_time = time.time()

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    end_time = time.time()
    print(f"RUNTIME OF BASELINE RANDOM FOREST CLASSIFIER: {end_time - start_time:.2f} SECONDS")

    importances = rf.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # sort by importance in decending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    # print(y_pred)

    # print("Accuracy of Random Forest 1: " + str(rf.score(X_test, y_test)))
    print("Accuracy of Random Forest 1 (all features): " + str(accuracy_score(y_test, y_pred)))
    print("F1 Score of Random Forest 1 (all features): " + str(f1_score(y_test, y_pred)))
    
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'max_features': ['sqrt'],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        n_iter=10,
        scoring='f1_weighted',
        cv=3,
        n_jobs=-1
    )
    print(search.get_params())

    # generate Random Forest classifier with RandomizedSearchCV and calculate runtime
    start_time = time.time()

    search.fit(X_train, y_train)

    end_time = time.time()
    print(f"RUNTIME OF OPTIMIZED RANDOM FOREST CLASSIFIER USING RANDOMIZEDSEARCHCV: {end_time - start_time:.2f} SECONDS")

    model = search.best_estimator_

    y_pred_best = model.predict(X_test)

    print("Accuracy of Random Forest 2 (best model): " + str(accuracy_score(y_test, y_pred_best)))
    print("F1 Score of Random Forest 2 (best model): " + str(f1_score(y_test, y_pred_best)))


if __name__ == "__main__":
    random_forest()


# resources used as guidance
# - https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/
# - https://www.youtube.com/watch?v=_QuGM_FW9eo