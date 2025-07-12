# python file for analyzing and visualizing the dataset
# spotting trends, patterns, or anomalies 

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

def load_data(path=".../data/student_depression_dataset.csv"):
    return pd.read_csv(path)

def target_distribution(df, target_column="Depression"):
    sns.countplot(x=target_column, data=df)
    plt.title("Depression Distribution")
    plt.show()

# When you close the target distribution window, the plot below will appear.
def cgpa_distribution(df, bins=20):
    sns.histplot(data=df, x="CGPA", hue="Depression", bins=bins, multiple="dodge", hue_order=[0,1])
    plt.xticks(range(0,11))
    plt.title("CGPA Distribution by Depression Status")
    plt.xlabel("CGPA")
    plt.ylabel("Number of Students")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    target_distribution(df)
    cgpa_distribution(df, bins=20)
