# baseline model python file

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from evaluation import plot_cf

def load_data():
    return pd.read_csv("../data/student_depression_dataset.csv")

def train_baseline_model():
    df = load_data()

    # preprocess data and drop certain columns that may affect accuracy of model
    drop_cols = ['id', 'Work Pressure', 'Job Satisfaction', 'Gender', 'City', 'Profession']
    df = df.drop(columns=drop_cols)
    print(df.columns)

    # encode categorical variables
    df_encoded = pd.get_dummies(df)

    # variable X contains all non-dropped columns except for 'Depression'
    X = df_encoded.drop("Depression", axis=1)

    # y variable contains only the 'Depression' column to serve as the target label the model tries to predict
    y = df_encoded["Depression"]

    # using train_test_split from sklearn.model_selection to split dataset into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 99)

    # define TreeClassifier model
    clf = DecisionTreeClassifier(random_state=1)

    # train the model
    clf.fit(X_train, y_train)

    # make predictions
    y_pred = clf.predict(X_test)

    # prints accuracy score of actual vs. predicted
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    return y_test, y_pred

if __name__ == "__main__":
    y_test, y_pred = train_baseline_model()

    labels = ['Not Depressed', 'Depressed']
    plot_cf(y_test, y_pred, labels)


# resources used as guidance
# - https://www.geeksforgeeks.org/machine-learning/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/
# - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# - https://www.geeksforgeeks.org/pandas/python-pandas-get_dummies-method/ 
# - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html