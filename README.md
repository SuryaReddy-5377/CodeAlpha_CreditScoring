
---

## 📂 TASK 3: Credit Scoring Model

**File:** `README.md`

```markdown
# Credit Scoring Model

## 📌 Project Overview
This project predicts an individual's creditworthiness using past financial data. The model helps financial institutions decide whether to approve or reject loan applications.

## 📊 Results
- **Best Model:** Random Forest
- **Accuracy:** ~74%
- **Precision:** ~73%
- **Recall:** ~68%
- **F1-Score:** ~70%

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 70.00% | 0.72 | 0.68 | 0.70 |
| Decision Tree | 66.50% | 0.67 | 0.65 | 0.66 |
| Random Forest | **74.00%** | **0.73** | **0.68** | **0.70** |
| SVM | 68.50% | 0.70 | 0.66 | 0.68 |

## 🛠️ Technologies Used
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## 📂 Dataset
**German Credit Data (UCI ML Repository)**
- Source: UCI Machine Learning Repository
- 1,000 records
- 20 features
- Target: 0 = Good Credit, 1 = Bad Credit

### Features Used
- status - Current account status
- duration - Loan duration in months
- credit_history - Credit history
- purpose - Purpose of loan
- amount - Credit amount
- savings - Savings account/bonds
- employment - Present employment since
- installment_rate - Installment rate as % of income
- personal_status - Personal status and sex
- other_debtors - Other debtors/guarantors
- residence - Present residence since
- property - Property
- age - Age in years
- other_installment - Other installment plans
- housing - Housing
- existing_credits - Number of existing credits at this bank
- job - Job
- liable_dependents - Number of people liable to provide maintenance
- telephone - Telephone
- foreign_worker - Foreign worker

## 🚀 How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the code
python credit_scoring.py
