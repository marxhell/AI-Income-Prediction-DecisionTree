import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Improved Machine Learning Model Starting...\n")

# Load dataset
df = pd.read_csv("adult.csv")

# Column names
df.columns = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","gender","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

# Replace missing values
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

print("Dataset cleaned\n")

# Encode categorical columns
label_encoder = LabelEncoder()

categorical_columns = [
    "workclass","education","marital-status","occupation",
    "relationship","race","gender","native-country","income"
]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Encoding complete\n")

# Feature selection (remove noisy features)
X = df.drop(["income","fnlwgt"], axis=1)
y = df["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset split\n")

# Improved Decision Tree
model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model training complete\n")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("MODEL PERFORMANCE")
print("------------------")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
