# **Toxicity Classification Workflow**
* This project builds a **machine learning pipeline** to classify compounds as Toxic or NonToxic using QSAR molecular descriptors.

* The dataset is provided in **Toxicity Dataset.csv**, with the target column named Class (renamed to class in code for consistency).

# **1. Data loading and basic EDA**
* The notebook starts by loading the CSV into a pandas DataFrame and renaming the target column to class:

python:
```
df = pd.read_csv('Toxicity Dataset.csv')
df = df.rename(columns={'Class': 'class'})
A quick exploratory analysis is performed:

df.head() to visually inspect the first few rows.
df.describe() to summarize numeric features.
df['class'].value_counts() to inspect class balance.
df.isna().sum() to check for missing values.
```
**Result:**
* The dataset has 171 samples and 1200+ descriptor columns. The classes are moderately imbalanced (roughly 67 percent NonToxic, 33 percent Toxic), and all features are numeric descriptors with no substantial missing-value issues.

# **2. Preprocessing**
Features (X) and target (y) are separated:

python:
```
X = df.drop(columns=['class'])
y = df['class']
```
* Columns are split into numeric and categorical (in this dataset, all are numeric):

python:
```
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
```
* A preprocessing pipeline is defined:

**Numeric features**
Median imputation (safety net for any missing values) + standardization.

**Categorical features (if any exist)**
Most-frequent imputation + one-hot encoding.

python:
```
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)
```
# **3. Feature selection**
* Because the dataset is high-dimensional (1200+ features) and low-sample (171 rows), feature selection is added to reduce overfitting and improve interpretability.

* A RandomForestClassifier is used as an embedded feature selector:

python:
```
fs_estimator = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

feature_selector = SelectFromModel(
    estimator=fs_estimator,
    threshold='median',   # keep features with importance above the median
    prefit=False
)
```
* This step runs after preprocessing in the pipeline: the forest is fit on the transformed features, and only the more important descriptors are passed to the final classifier.

# **4. Model and full pipeline**
* The base classifier is another RandomForestClassifier (you can easily swap this for another ensemble model):

python:
```
rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
The full pipeline combines:

Preprocessing (preprocess)
Feature selection (feature_selector)
Final model (rf_clf)
```

python:
```
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('feature_selection', feature_selector),
    ('model', rf_clf)
])
```
* This design ensures that all preprocessing and feature-selection steps are fit only on the training data inside cross-validation, avoiding data leakage and making the workflow reproducible.

# **5. Train/test split and cross-validation**
* The data is split into train and test sets with stratification on the target to preserve class proportions:

python:
```
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```
* Cross-validation is done only on the training set using StratifiedKFold and macro F1 as the main metric (to balance performance across both classes):

python:
```
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

print("CV F1-macro scores:", cv_scores)
print("Mean CV F1-macro:", cv_scores.mean())
```
* This provides a robust estimate of generalization performance before touching the held-out test set.

# **6. Final training and evaluation**
* After cross-validation, the pipeline is trained on the full training set and evaluated on the held-out test set:

python:
```
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
**Typical outputs include:**

* Classification report
Precision, recall, and F1-score for both Toxic and NonToxic.

* Confusion matrix
How many samples of each class are correctly/incorrectly classified.

Optionally, the confusion matrix is visualized:

python:
```
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=pipe.classes_,
            yticklabels=pipe.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```
# **7. Inspecting selected features (optional)**
* After fitting, the selected feature indices can be extracted and mapped back to the original descriptor names (for interpretability):

python:
```
# Get boolean mask of selected features after preprocessing
selector = pipe.named_steps['feature_selection']
support_mask = selector.get_support()

# Get transformed feature names from the ColumnTransformer
# (for pure-numeric QSAR data, this corresponds directly to num_cols)
selected_feature_names = np.array(num_cols)[support_mask]

print("Number of selected features:", support_mask.sum())
print("Selected features (first 20):", selected_feature_names[:20])
```
* This gives insight into which molecular descriptors are most relevant for distinguishing between Toxic and NonToxic compounds.
