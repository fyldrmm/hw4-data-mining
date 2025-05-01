import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("2020.csv")

df['Happiness Level'] = pd.qcut(df['Happiness score'], q=3, labels=['Low', 'Medium', 'High'])
features = df.drop(columns=['Country name', 'Happiness Rank', 'Happiness score', 'Happiness Level'])
features.columns = features.columns.str.strip()
target = df['Happiness Level']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Feature Importances:\n", pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False))