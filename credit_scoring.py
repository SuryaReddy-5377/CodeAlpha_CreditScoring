"""
TASK 3: Credit Scoring Model - WORKING VERSION
CodeAlpha Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("💰 CREDIT SCORING MODEL - TASK 3")
print("="*60)

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n📥 Loading German Credit Data...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ['status', 'duration', 'credit_history', 'purpose', 'amount',
           'savings', 'employment', 'installment_rate', 'personal_status',
           'other_debtors', 'residence', 'property', 'age', 'other_installment',
           'housing', 'existing_credits', 'job', 'liable_dependents',
           'telephone', 'foreign_worker', 'credit_risk']

df = pd.read_csv(url, sep=' ', names=columns)

# Convert target: 1=Good, 2=Bad -> 0=Good, 1=Bad
df['credit_risk'] = df['credit_risk'].replace({1: 0, 2: 1})

print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================
# STEP 2: CONVERT STRING TO NUMBERS (FIX)
# ============================================
print("\n🔄 Converting string columns to numbers...")

label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✅ {col}: Converted to numbers")

print("\n📊 Data Types After Conversion:")
print(df.dtypes)

# ============================================
# STEP 3: PREPARE DATA
# ============================================
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

print(f"\n📊 Class Distribution:")
print(f"  Good Credit (0): {(y==0).sum()}")
print(f"  Bad Credit (1): {(y==1).sum()}")

# ============================================
# STEP 4: SPLIT & SCALE
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Training: {len(X_train)} samples")
print(f"✅ Testing: {len(X_test)} samples")

# ============================================
# STEP 5: TRAIN MODELS
# ============================================
print("\n" + "="*60)
print("🚀 TRAINING MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

results = {}

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"  ✅ Accuracy: {accuracy*100:.2f}%")
    print(f"     Precision: {precision*100:.2f}%")
    print(f"     Recall: {recall*100:.2f}%")
    print(f"     F1-Score: {f1*100:.2f}%")

# ============================================
# STEP 6: RESULTS
# ============================================
print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")

print("\n📊 All Models Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall: {metrics['recall']*100:.2f}%")
    print(f"  F1-Score: {metrics['f1']*100:.2f}%")

# ============================================
# STEP 7: CONFUSION MATRIX
# ============================================
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)
print(f"\n📊 Confusion Matrix - {best_model_name}:")
print(f"  True Negatives (Good Credit): {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives (Bad Credit): {cm[1][1]}")

# ============================================
# STEP 8: VISUALIZE
# ============================================
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Good Credit', 'Bad Credit'])
plt.yticks(tick_marks, ['Good Credit', 'Bad Credit'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), horizontalalignment="center", 
                 color="white" if cm[i][j] > cm.max()/2 else "black")

plt.tight_layout()
plt.show()

# ============================================
# STEP 9: TEST SAMPLE
# ============================================
print("\n" + "="*60)
print("🔍 TEST ON SAMPLE CLIENT")
print("="*60)

sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)

if hasattr(best_model, 'predict_proba'):
    prob = best_model.predict_proba(sample_scaled)[0]
    print(f"\n📊 Prediction Probabilities:")
    print(f"  Good Credit: {prob[0]*100:.1f}%")
    print(f"  Bad Credit: {prob[1]*100:.1f}%")

if prediction[0] == 0:
    print(f"\n✅ DECISION: APPROVED (Good Credit)")
else:
    print(f"\n❌ DECISION: REJECTED (Bad Credit)")

print("\n" + "="*60)
print("🎉 TASK 3 COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
