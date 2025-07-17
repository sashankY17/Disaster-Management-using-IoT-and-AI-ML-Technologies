import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("sensor_data_labeled_95_accuracy.csv")

# Features and target
X = df[['temperature', 'humidity', 'gas', 'flame', 'water_level', 'vibration']]
y = df['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train a model (Random Forest here, but you can replace it with any classifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Disaster"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Disaster Detection")
plt.grid(False)
plt.show()
