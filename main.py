import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path):
    df = pd.read_csv(path)
    print("First 5 rows:\n", df.head())
    print("\nInfo:\n")
    print(df.info())
    print("\nClass Distribution:\n", df['Class'].value_counts())
    return df

def visualize_distribution(df):
    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
    plt.show()

def preprocess_and_split(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Show model accuracy
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    
    return model


def save_model(model, scaler):
    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Model and scaler saved.")

def feature_importance_plot(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Top Features")
    plt.bar(range(10), importances[indices[:10]], align="center")
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
    plt.tight_layout()
    plt.show()

def predict_sample():
    model = joblib.load("fraud_model.pkl")
    sample = pd.DataFrame([{
        "Time": 0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }])
    prediction = model.predict(sample)
    print("\nSample Prediction:", "Fraud" if prediction[0] == 1 else "Legit")

def main():
    df = load_data("fraud_dataset_sample.csv")
    visualize_distribution(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(df)
    model = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_model(model, scaler)
    feature_importance_plot(model, feature_names)
    predict_sample()

if __name__ == "__main__":
    main()
