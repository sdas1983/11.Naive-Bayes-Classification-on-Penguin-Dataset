# Naive Bayes Classification on Penguin Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
import pickle

# Setting display options for pandas
pd.set_option('display.max_columns', None)

# Load the dataset
file_path = r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\penguins_size.csv"
df = pd.read_csv(file_path)
df = df.drop('Unnamed: 0', axis=1)

# Initial Data Exploration
df.info()
df.isnull().sum()

# Dropping rows with missing values
df = df.dropna()
df.isnull().sum()

# Handling Categorical Variables
# Exploring unique values
df['island'].unique()
df['sex'].unique()

# There is one entry in 'sex' with a value '.'
# We will assign it based on descriptive statistics
df[df['sex'] == '.']
df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose()
df.at[336, 'sex'] = 'FEMALE'

# Verifying the correction
df['sex'].unique()

# Data Visualization
sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', data=df, hue='species', palette='Dark2')
sns.pairplot(df, hue='species')
sns.catplot(x='species', y='culmen_length_mm', col='sex', data=df, kind='box')

# Encoding Categorical Variables
df['sex'] = df['sex'].map({'FEMALE': 0, 'MALE': 1})
df['island'] = df['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})

# Separating Features and Target Variable
X = df.drop(columns=['species'])
y = df['species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.35, random_state=30)

# Naive Bayes Model
naive_bayes = GaussianNB()

# Training the Model
naive_bayes.fit(X_train, y_train)

# Making Predictions
y_pred = naive_bayes.predict(X_test)

# Model Evaluation
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print(f'Training Score: {naive_bayes.score(X_train, y_train)}')
print(f'Testing Score: {naive_bayes.score(X_test, y_test)}')

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(naive_bayes, X_test, y_test)

# ROC Curve Display
# RocCurveDisplay.from_estimator(naive_bayes, X_test, y_test)

# Multiclass ROC AUC Curve
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(8,8)):
    y_score = clf.predict_proba(X_test)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    # One-vs-all ROC calculation
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plotting the ROC curve for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curve')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'ROC Curve (area = {roc_auc[i]:.2f}) for label {i}')
    ax.legend(loc="best")
    ax.grid(alpha=0.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(naive_bayes, X_test, y_test, n_classes=3)

# Saving the Model
with open('naive_bayes_model.pickle', 'wb') as model_file:
    pickle.dump(naive_bayes, model_file)

# Loading the Model
with open('naive_bayes_model.pickle', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Predicting with a Test Input
test_input = scaler.transform([[0, 40, 20, 190, 3800, 1]])
print(f'Predicted Class: {loaded_model.predict(test_input)[0]}')
