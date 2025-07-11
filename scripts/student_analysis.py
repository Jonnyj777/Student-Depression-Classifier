# python file for analyzing and visualizing the dataset
# spotting trends, patterns, or anomalies 

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

def load_data(path="../data/student_depression_dataset.csv"):
    return pd.read_csv(path)

def target_distribution(df, target_column="Depression"):
    sns.countplot(x=target_column, data=df)
    plt.title("Depression Distribution")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    target_distribution(df)
