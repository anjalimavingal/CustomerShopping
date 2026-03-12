import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv('data/online vs store shopping dataset.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().any())

#EDA

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="shopping_preference")
plt.title("Distribution of Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="age")
plt.title("Age vs Shopping Preference")
plt.show()


plt.figure(figsize=(6,4))
sns.violinplot(data=df, x="shopping_preference", y="monthly_income")
plt.title("Monthly income vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="daily_internet_hours")
plt.title("Daily internet vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="smartphone_usage_years")
plt.title("Smartphone usage vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="social_media_hours")
plt.title("Social media hours vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="online_payment_trust_score")
plt.title("Online Payment Trust score vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="tech_savvy_score")
plt.title("Comfort with technology score vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="product_availability_online")
plt.title("Perceived online product availability vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="impulse_buying_score")
plt.title("Likelihood of impulse purchases vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="need_touch_feel_score")
plt.title("Preference to see/touch products before buying vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="brand_loyalty_score")
plt.title("Brand loyalty level vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="environmental_awareness")
plt.title("Eco-consciousness level vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="shopping_preference", y="time_pressure_level")
plt.title("Perceived lack of time vs Shopping Preference")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(data=df, x="gender", hue="shopping_preference")
plt.title("Gender vs Shopping Preference")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="city_tier", hue="shopping_preference")
plt.title("City Tier vs Shopping Preference")
plt.show()

plt.figure(figsize=(16,10))
sns.heatmap(df.drop(['gender','city_tier','shopping_preference'], axis=1).corr(),
            cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

#outlier treatment ( IQR capping)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
    

# # feature engineering setup
X = df.drop("shopping_preference", axis=1)
y = df["shopping_preference"]

categorical_cols = ["gender", "city_tier"]
numeric_cols = X.select_dtypes(include=np.number).columns

#Preprocessing pipeline
#include PowerTransformer(skewness reduction), StandardScaler, OneHotEncoding, PCA

numeric_pipeline = Pipeline([
    ("power", PowerTransformer()),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

pca = PCA(n_components=0.95, random_state=42)

#train test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#model building
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\nModel: {name}")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.show()
    
    return acc, f1

#Logistic regression
log_model = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

log_acc, log_f1 = evaluate_model(log_model, X_test, y_test, "Logistic Regression")


#Random forest
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

rf_model.fit(X_train, y_train)

rf_acc, rf_f1 = evaluate_model(rf_model, X_test, y_test, "Random Forest")

#Support Vector machine
svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("classifier", SVC(kernel="rbf", probability=True))
])

svm_model.fit(X_train, y_train)

svm_acc, svm_f1 = evaluate_model(svm_model, X_test, y_test, "SVM")

#Gradient boosting

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("classifier", GradientBoostingClassifier())
])

gb_model.fit(X_train, y_train)

gb_acc, gb_f1 = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

#GDecision Tree

dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("classifier", DecisionTreeClassifier(max_depth=4))
])

dt_model.fit(X_train, y_train)

dt_acc, dt_f1 = evaluate_model(dt_model, X_test, y_test, "Decision Tree")


#Model Comparison
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "Decision Tree"],
    "Accuracy": [log_acc, rf_acc, svm_acc, gb_acc,dt_acc],
    "F1 Score": [log_f1, rf_f1, svm_f1, gb_f1,dt_f1]
})

print(results.sort_values(by="F1 Score", ascending=False))

joblib.dump(log_model, "model.pkl")