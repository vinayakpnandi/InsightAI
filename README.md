
# InsightAI 🚀

### AI-Powered Data Analytics & Decision Intelligence Platform

InsightAI is a Streamlit-based data analytics platform that combines traditional data analysis, SQL, machine learning, visualization, prediction, and local LLM-powered business insights into a single application.

The goal is to allow users to upload a dataset, analyze it, ask questions in natural language, build ML models, make predictions, and receive business-oriented insights.

---

## 🎯 Current Capabilities

### 📊 Dataset Profiling

InsightAI automatically profiles uploaded datasets and provides:

- Number of rows
- Number of columns
- Numeric columns
- Categorical columns
- Missing-value analysis
- Duplicate-row detection
- Basic dataset statistics

---

### 🧹 Data Quality Analysis

The platform performs basic data-quality checks before analysis, helping identify issues such as:

- Missing values
- Duplicate records
- Data-type information
- Dataset structure

---

### 🗄️ Local Database Integration

Uploaded datasets are loaded into a local SQLite database.

The current database workflow is:

```text
CSV / Dataset
     ↓
Dataset Loader
     ↓
SQLite
     ↓
uploaded_data
     ↓
SQL Analysis
````

---

### 💬 Natural-Language SQL

InsightAI includes an SQL agent that converts natural-language business questions into SQL queries.

Example:

```text
Show revenue by country
```

The system generates a query such as:

```sql
SELECT Country, SUM(Revenue) AS total_revenue
FROM uploaded_data
GROUP BY Country;
```

The generated SQL is validated before execution to ensure that only safe, read-only queries are executed.

---

### 📈 Automated Visualization

InsightAI can automatically generate visualizations from the uploaded dataset.

The visualization module uses Python-based visualization tools to help users understand:

* Trends
* Comparisons
* Distributions
* Relationships between variables
* Category-level performance

---

# 🤖 Machine Learning

InsightAI includes an automated ML pipeline.

The pipeline performs:

```text
Dataset
   ↓
Target Selection
   ↓
Problem Detection
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model
   ↓
Explainability
```

---

## 🔍 Automatic Problem Detection

The system automatically determines whether the selected target is a:

### Classification problem

For categorical targets such as:

```text
Product_Category
Customer_Gender
Age_Group
```

### Regression problem

For numeric targets such as:

```text
Revenue
Profit
Cost
Price
```

The system was also updated to recognize **currency-formatted string values** such as:

```text
"$2,320.00"
"$1,043.00"
"$1,054.00"
```

and convert them into numeric values for regression.

---

## 📐 Model Evaluation

### Classification

Classification models are evaluated using:

* Accuracy

### Regression

Regression models are evaluated using:

* R²
* MAE
* RMSE

The system compares candidate models and selects the best-performing model.

---

## 🔎 Feature Importance

InsightAI provides feature-importance information for the trained model.

This allows users to understand which features have the strongest predictive relationship with the target.

Feature importance is treated as a **predictive signal, not causal evidence**.

---

## ⚠️ Data Leakage Detection

InsightAI includes target-leakage analysis.

For example, while analyzing a Revenue target, the system can detect relationships such as:

```text
Revenue = Unit Price × Order Quantity
```

and:

```text
Revenue = Cost + Profit
```

These relationships can create artificially high model performance.

InsightAI therefore reports potential leakage risks alongside the ML results.

---

# 🎯 Interactive Prediction

After training a model, InsightAI provides an interactive prediction interface.

The prediction system supports both:

### Classification

```text
Input Features
      ↓
Trained Classifier
      ↓
Predicted Class
```

### Regression

```text
Input Features
      ↓
Trained Regressor
      ↓
Predicted Numeric Value
```

For example:

```text
Predicted Revenue
$2,487.35
```

---

# 🧠 AI Business Insights

InsightAI uses a local Qwen3 model through Ollama to convert ML results into a structured business report.

The generated report contains:

### Executive Summary

A concise overview of the ML analysis.

### Key Insights

Important patterns identified from the analysis.

### Risks

Potential issues such as data leakage.

### Recommendations

Suggested actions based on the analysis.

### Prediction Interpretation

A business-friendly explanation of the prediction.

Example structure:

```text
EXECUTIVE SUMMARY

Random Forest achieved strong predictive performance,
but medium data-leakage risk was detected.

KEY INSIGHTS

• Order Quantity is an important predictive feature.
• Cost and Profit contain predictive signals.
• Model performance should be interpreted alongside
  leakage risk.

RISKS

• Revenue may be derived from other variables.
• Model performance may therefore be overstated.

RECOMMENDATIONS

• Validate feature engineering.
• Review leakage-prone variables.
• Re-evaluate model performance.
```

---

# 🧠 Local LLM Architecture

InsightAI currently uses **Ollama + Qwen3:4b** for its AI-powered functionality.

The current architecture includes separate LLM usage for:

```text
                 Ollama
                   │
                Qwen3:4b
                   │
          ┌────────┴────────┐
          │                 │
      SQL Agent      Business Insights
          │                 │
      SQL Query       Business Report
```

The model runs locally through Ollama.

---

# 🧩 Project Architecture

```text
InsightAI/
│
├── app.py
│
├── src/
│   │
│   ├── database/
│   │   ├── database.py
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── sql_agent.py
│   │
│   ├── ml/
│   │   ├── profiler.py
│   │   ├── predictor.py
│   │   ├── prediction.py
│   │   ├── models.py
│   │   ├── preprocessor.py
│   │   ├── explainability.py
│   │   ├── leakage.py
│   │   ├── service.py
│   │   └── business_insights.py
│   │
│   ├── rag/
│   │
│   ├── tools/
│   │   ├── data_quality.py
│   │   └── visualizer.py
│   │
│   └── utils/
│       └── llm.py
│
├── tests/
│
├── data/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Core development        |
| Streamlit    | Web application         |
| Pandas       | Data processing         |
| NumPy        | Numerical operations    |
| Scikit-learn | Machine learning        |
| Matplotlib   | Visualization           |
| SQLite       | Local database          |
| LangChain    | LLM integration         |
| Ollama       | Local LLM inference     |
| Qwen3:4b     | AI generation           |
| ChromaDB     | Vector-store components |
| Git          | Version control         |
| GitHub       | Source control          |

---

# 🧪 Testing

The project contains tests for several major components:

```text
tests/
│
├── test_dataset.py
├── test_ml.py
├── test_ml_service.py
├── test_prediction.py
├── test_explainability.py
├── test_business_insights.py
├── test_business_llm.py
├── test_sql_agent.py
├── test_ingestion.py
├── test_retrieval.py
├── test_qa.py
└── check_database.py
```

The ML pipeline and Business Intelligence pipeline have been tested successfully with the project datasets.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/vinayakpnandi/InsightAI.git
```

```bash
cd InsightAI
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# 🧠 Ollama Setup

Install Ollama and pull the Qwen3 model:

```powershell
ollama pull qwen3:4b
```

Verify:

```powershell
ollama list
```

The application expects Ollama to be available at:

```text
http://localhost:11434
```

---

# 🔐 Environment Variables

Create a `.env` file using `.env.example` as a reference.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_BUSINESS_MODEL=qwen3:4b
OLLAMA_SQL_MODEL=qwen3:4b
```

The actual `.env` file is excluded from GitHub.

---

# ▶️ Run the Application

Start InsightAI using:

```powershell
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

---

# 🚀 Current Workflow

The current working InsightAI pipeline is:

```text
Upload Dataset
      ↓
Dataset Profiling
      ↓
Data Quality
      ↓
SQLite Database
      ↓
Natural-Language SQL
      ↓
Visualization
      ↓
ML Problem Detection
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Leakage Detection
      ↓
Interactive Prediction
      ↓
AI Business Insights
```

---

## 👨‍💻 Author

**Vinayak Prakash Nandi**

Data Science Engineering Student
