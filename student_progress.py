import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(r'D:\PROJECT\student_data.csv')

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male=1, Female=0
data['Participation'] = le.fit_transform(data['Participation'])  # High=2, Medium=1, Low=0
data['Performance'] = le.fit_transform(data['Performance'])  # Pass=1, Fail=0

X = data[['Gender', 'StudyHours', 'PreviousScores', 'Participation']]
y = data['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("🔹 Model Accuracy:", accuracy)
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Student Performance Prediction")
plt.show()
