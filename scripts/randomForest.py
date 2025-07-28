# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("Starting Random Forest")

    df = pd.read_csv("data/student_depression_dataset.csv", encoding="latin-1")


    df = df.drop(columns=['Gender'])

    # Encode categorical variables
    df = pd.get_dummies(df)

    # Drop rows with missing values (optional but often necessary)
    df = df.dropna()

    # print the shape of the dataframe
    print(df.shape)

    # print the columns of the dataframe
    print(df.columns)

    X = df.iloc[:, 0:13] #???
    y = df.iloc[:, 13]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print(y_pred)

    print(rf.score(X_test, y_test))

    print(classification_report(y_test, y_pred))

    features = pd.DataFrame(rf.feature_importances_, index=X.columns)

    print(features) # Take a look at the features and their importance

    rf2 = RandomForestClassifier(n_estimators=1000, 
                                criterion='entropy',
                                max_depth=14,
                                min_samples_split=10,
                                random_state=17)

    rf2.fit(X_train, y_train)

    print(rf2.score(X_test, y_test))

    y_pred2 = rf2.predict(X_test)   
    print(classification_report(y_test, y_pred2))


if __name__ == "__main__":
    main()
