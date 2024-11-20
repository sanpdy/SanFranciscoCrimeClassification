import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
train = pd.read_csv(r"C:\Users\sanka\sanfran\data\train.csv")
test = pd.read_csv(r"C:\Users\sanka\sanfran\data\test.csv")

# 1. Data Exploration and Visualization
# ----------------------------------------------------
# Create directories to save the plots if they don't exist
import os
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Crimes per Year
plt.figure(figsize=(10,6))
train['Year'] = pd.to_datetime(train['Dates']).dt.year
sns.countplot(data=train, x='Year', order=sorted(train['Year'].unique()))
plt.title('Number of Crimes per Year')
plt.xlabel('Year')
plt.ylabel('Crime Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'crimes_per_year.png'))
plt.close()

# Crimes per Month
plt.figure(figsize=(10,6))
train['Month'] = pd.to_datetime(train['Dates']).dt.month
sns.countplot(data=train, x='Month')
plt.title('Number of Crimes per Month')
plt.xlabel('Month')
plt.ylabel('Crime Count')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'crimes_per_month.png'))
plt.close()

# Crimes per Day of Week
plt.figure(figsize=(10,6))
sns.countplot(data=train, x='DayOfWeek', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Number of Crimes per Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Crime Count')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'crimes_per_day_of_week.png'))
plt.close()

# Crimes per Hour
plt.figure(figsize=(10,6))
train['Hour'] = pd.to_datetime(train['Dates']).dt.hour
sns.countplot(data=train, x='Hour')
plt.title('Number of Crimes per Hour')
plt.xlabel('Hour')
plt.ylabel('Crime Count')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'crimes_per_hour.png'))
plt.close()

# Heatmap of Crimes over Time (Day of Week vs Hour)
pivot = train.pivot_table(index='DayOfWeek', columns='Hour', values='Category', aggfunc='count')
plt.figure(figsize=(12,6))
sns.heatmap(pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']), cmap='viridis')
plt.title('Heatmap of Crimes (Day of Week vs Hour)')
plt.xlabel('Hour')
plt.ylabel('Day of Week')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'heatmap_crimes_day_hour.png'))
plt.close()

# 2. Feature Engineering
# ----------------------------------------------------
def process_data(df):
    # Convert dates to datetime and extract features
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Day'] = df['Dates'].dt.day
    df['Hour'] = df['Dates'].dt.hour
    df['Minute'] = df['Dates'].dt.minute
    df['DayOfWeek_num'] = df['Dates'].dt.dayofweek  # Monday=0
    df['IsWeekend'] = df['DayOfWeek_num'].isin([5,6]).astype(int)
    df['Quarter'] = df['Dates'].dt.quarter
    df['WeekOfYear'] = df['Dates'].dt.isocalendar().week

    # Process 'Address' field
    df['Intersection'] = df['Address'].apply(lambda x: 1 if '/' in x else 0)

    # Encode 'PdDistrict' using Label Encoding
    df['PdDistrict'] = df['PdDistrict'].astype('category')

    # Keep necessary columns
    features = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek_num',
                'IsWeekend', 'Quarter', 'WeekOfYear', 'PdDistrict',
                'Intersection', 'X', 'Y']
    return df[features]

# Process datasets
X_train = process_data(train)
X_test = process_data(test)
y_train = train['Category']

# Encode target variable
le_y = LabelEncoder()
y_train_encoded = le_y.fit_transform(y_train)

# Identify categorical features
cat_features = ['PdDistrict', 'Intersection', 'DayOfWeek_num', 'IsWeekend', 'Quarter']

# 3. Model Training with Cross-Validation and Metrics Logging
# -----------------------------------------------------------
# Cross-Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store predictions and metrics
test_preds = np.zeros((len(X_test), len(le_y.classes_)))
oof_preds = np.zeros((len(X_train), len(le_y.classes_)))
accuracy_list = []
f1_list = []
classification_reports = []
confusion_matrices = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
    print(f"\nFold {fold + 1}")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        eval_metric='MultiClass',
        task_type='GPU',
        cat_features=cat_features,
        early_stopping_rounds=50,
        random_seed=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=100
    )

    # Predict on validation set
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = model.predict_proba(X_val)

    # Calculate Metrics
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds, average='weighted')
    report = classification_report(y_val, val_preds, target_names=le_y.classes_)
    cm = confusion_matrix(y_val, val_preds)

    accuracy_list.append(acc)
    f1_list.append(f1)
    classification_reports.append(report)
    confusion_matrices.append(cm)

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    print(f"Fold {fold + 1} F1 Score: {f1:.4f}")

    # Predict on test set
    test_preds += model.predict_proba(X_test) / skf.n_splits

# Overall Metrics
print("\nOverall Performance:")
print(f"Mean Accuracy: {np.mean(accuracy_list):.4f}")
print(f"Mean F1 Score: {np.mean(f1_list):.4f}")

# 4. Confusion Matrix Visualization
# ----------------------------------------------------
# Since the number of classes is large, select the top N classes for visualization
N = 10  # Number of classes to display
class_counts = np.bincount(y_train_encoded)
top_N_classes = np.argsort(class_counts)[-N:]

# Create a mapping from old labels to new labels
label_mapping = {old_label: new_label for new_label, old_label in enumerate(top_N_classes)}

# Filter confusion matrix for top N classes
cm_top_N = confusion_matrices[-1][top_N_classes][:, top_N_classes]

# Normalize confusion matrix
cm_normalized = cm_top_N / cm_top_N.sum(axis=1, keepdims=True)

# Plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le_y.inverse_transform(top_N_classes),
            yticklabels=le_y.inverse_transform(top_N_classes))
plt.title('Normalized Confusion Matrix - Fold {}'.format(fold + 1))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'confusion_matrix_top_N.png'))
plt.close()

# 5. Feature Importance
# ----------------------------------------------------
feature_importances = model.get_feature_importance()
feature_names = X_train.columns

fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
fi_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(data=fi_df, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
plt.close()

# 6. Additional Visualizations
# ----------------------------------------------------

# 6.1 Crime Categories Distribution
plt.figure(figsize=(12,6))
crime_counts = train['Category'].value_counts()
sns.barplot(y=crime_counts.index[:15], x=crime_counts.values[:15])
plt.title('Top 15 Crime Categories')
plt.xlabel('Number of Occurrences')
plt.ylabel('Crime Category')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'top_15_crime_categories.png'))
plt.close()

# 6.2 Geographic Distribution of Crimes
plt.figure(figsize=(10,10))
sampled_data = train.sample(10000)
sns.scatterplot(x='X', y='Y', data=sampled_data, hue='Category', legend=False, s=10)
plt.title('Geographic Distribution of Crimes (Sample of 10,000)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'geographic_distribution.png'))
plt.close()

# 6.3 Crimes Over Time
plt.figure(figsize=(12,6))
train['Date'] = train['Dates'].dt.date
daily_crimes = train.groupby('Date').size()
plt.plot(daily_crimes.index, daily_crimes.values)
plt.title('Crimes Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'crimes_over_time.png'))
plt.close()

# 6.4 Crimes by Police District
plt.figure(figsize=(10,6))
sns.countplot(data=train, y='PdDistrict', order=train['PdDistrict'].value_counts().index)
plt.title('Crimes by Police District')
plt.xlabel('Number of Crimes')
plt.ylabel('Police District')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'crimes_by_district.png'))
plt.close()

# 7. Create Submission
# ----------------------------------------------------
submission = pd.DataFrame(test_preds, columns=le_y.classes_)
submission.insert(0, 'Id', test['Id'])

submission.to_csv('submission.csv', index=False)
