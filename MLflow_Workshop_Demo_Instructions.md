# MLflow Workshop Demo Instructions

## 🎯 Quick Demo Guide for Workshop Presenter

This document provides **step-by-step instructions** for demonstrating the live MLflow environment during your workshop presentation.

---

## 🚀 Pre-Demo Checklist

### ✅ Ensure MLflow Server is Running
```bash
# Check if server is active
curl http://localhost:5000/health

# If not running, start with:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &
```

### ✅ Verify Demo Data
- **4 Experiments** with **19 total runs**
- **3 Registered models**
- **50+ generated artifacts** (plots, reports, models)

---

## 🎬 Demo Flow - Key URLs to Open

### 1. **Main Dashboard** - Start Here
**URL**: http://localhost:5000

**What to Show:**
- Overview of all experiments
- Run counts for each experiment
- Clean, professional interface
- Quick experiment navigation

**Key Points:**
- "This is the main MLflow dashboard where data scientists can see all their experiments"
- "Notice we have 4 different experiments representing different ML projects"
- "Each experiment contains multiple runs with different parameters and configurations"

---

### 2. **Iris Classification Experiment** - Core ML Workflow
**URL**: http://localhost:5000/#/experiments/1

**What to Show:**
- **7 runs** with different Random Forest configurations
- Parameter columns (n_estimators, max_depth)
- Metric columns (accuracy, precision, recall, f1_score)
- Run comparison capabilities
- Tags and metadata organization

**Demo Actions:**
1. **Sort by accuracy** - Click on accuracy column header
2. **Select multiple runs** - Check boxes for 2-3 runs
3. **Click "Compare"** - Show side-by-side comparison

**Key Points:**
- "Here we tested different Random Forest configurations"
- "MLflow automatically tracks all parameters, metrics, and artifacts"
- "Notice how easy it is to compare performance across different runs"

---

### 3. **Individual Run Detail** - Deep Dive
**URL**: http://localhost:5000/#/experiments/1/runs/d3d244ad224f4009827591fd3f47b749

**What to Show:**
- Complete run information (parameters, metrics, duration)
- **Artifacts section** with multiple generated files:
  - Feature importance plot
  - Confusion matrix
  - Learning curves
  - Classification report
  - Predictions CSV
- Model information and signature
- Tags and metadata

**Demo Actions:**
1. **Click on artifacts** - Show the artifact browser
2. **Download a plot** - Click on `feature_importance_random_forest_5.png`
3. **Show model details** - Scroll to model section

**Key Points:**
- "Every run captures complete reproducibility information"
- "Artifacts include all generated plots, reports, and the trained model"
- "This eliminates the 'I can't reproduce that result' problem"

---

### 4. **Run Comparison View** - Model Selection
**URL**: Navigate from experiment view by selecting 2+ runs and clicking "Compare"

**What to Show:**
- Side-by-side parameter comparison
- Metric comparison with highlighting of differences
- Parallel coordinates plot showing parameter relationships
- Scatter plot views of metrics vs parameters

**Key Points:**
- "This comparison view helps teams quickly identify the best performing models"
- "You can see exactly which parameters led to better performance"
- "The parallel coordinates plot shows parameter relationships visually"

---

### 5. **Model Registry** - Production Management
**URL**: http://localhost:5000/#/models

**What to Show:**
- 3 registered models:
  - `iris_classifier`
  - `wine_classifier`
  - `tuned_wine_classifier`
- Model versions and stages
- Source run linking

**Demo Actions:**
1. **Click on a model name** - Show model detail page
2. **Show version information** - Demonstrate versioning concept
3. **Explain stage transitions** - None → Staging → Production → Archived

**Key Points:**
- "The Model Registry provides centralized model management"
- "Teams can promote models through different stages"
- "Full traceability from registered model back to training run"

---

### 6. **Model Comparison Study** - Algorithm Comparison
**URL**: http://localhost:5000/#/experiments/3

**What to Show:**
- 3 different algorithms on the same dataset:
  - Random Forest (100% accuracy)
  - Logistic Regression (100% accuracy)
  - SVM (75.93% accuracy)
- Direct performance comparison
- Training time differences

**Key Points:**
- "This experiment compares different algorithms on the wine classification dataset"
- "We can quickly see that Random Forest and Logistic Regression perform best"
- "Training time is also tracked - important for production considerations"

---

### 7. **Hyperparameter Tuning Demo** - Systematic Optimization
**URL**: http://localhost:5000/#/experiments/4

**What to Show:**
- 6 runs with different parameter combinations
- Training curves (loss over epochs) for each run
- Systematic parameter exploration
- Performance consistency across runs

**Demo Actions:**
1. **Click on a run** to show training curves
2. **Show step-by-step metrics** - Demonstrate epoch tracking
3. **Compare parameter effects**

**Key Points:**
- "This shows systematic hyperparameter optimization"
- "MLflow tracks training progress step-by-step"
- "Teams can see which parameters work best"

---

### 8. **California Housing Regression** - Different ML Task
**URL**: http://localhost:5000/#/experiments/2

**What to Show:**
- Regression metrics (MSE, RMSE, R², MAE)
- Different model types (Gradient Boosting, Linear Regression)
- Training curves showing loss decrease

**Key Points:**
- "MLflow works for any ML task - classification, regression, NLP, etc."
- "Different metrics are automatically tracked based on the task type"
- "Same interface works across all ML frameworks"

---

## 🎯 Key Demo Messages

### **Main Value Propositions:**

1. **Reproducibility**: "Never lose track of what worked and why"
2. **Collaboration**: "Teams can easily share and compare experiments"
3. **Organization**: "No more scattered notebooks and random result files"
4. **Production Ready**: "Direct path from experiment to production deployment"
5. **Framework Agnostic**: "Works with any ML library or framework"

---

## 🎤 Suggested Talking Points

### **Opening** (Main Dashboard)
"What you're seeing here is a complete MLOps experiment tracking system. We've run real ML experiments with actual data and models. This isn't just a demo - it's a fully functional MLflow environment."

### **During Experiment View**
"Notice how every experiment run is automatically organized with parameters, metrics, and artifacts. No more digging through folders trying to find which model performed best."

### **During Run Detail**
"This level of detail is automatic. Every plot, every model, every piece of metadata is captured without any additional effort from the data scientist."

### **During Model Registry**
"The Model Registry bridges the gap between experimentation and production. Models can be promoted through stages just like code deployments."

### **Closing**
"This is what modern MLOps looks like - systematic, reproducible, and production-ready from day one."

---

## ⚡ Quick Navigation Shortcuts

### **Essential URLs** (Bookmark These):
- **Main**: http://localhost:5000
- **Iris Exp**: http://localhost:5000/#/experiments/1
- **Models**: http://localhost:5000/#/models
- **Housing**: http://localhost:5000/#/experiments/2
- **Tuning**: http://localhost:5000/#/experiments/4

### **Backup Navigation:**
If URLs don't work, navigate manually:
1. Start at http://localhost:5000
2. Click experiment names to enter experiments
3. Click run names to see run details
4. Use "Models" tab for Model Registry

---

## 🛠️ Technical Backup Information

### **If MLflow Server Stops:**
```bash
cd /home01/harshvardhan.raju/mlops/kt
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### **Key Statistics to Mention:**
- **4 experiments** covering different ML use cases
- **19 total runs** with comprehensive tracking
- **3 registered models** ready for deployment
- **50+ artifacts** including plots, models, and reports
- **Multiple algorithms**: Random Forest, SVM, Logistic Regression, Gradient Boosting
- **Multiple datasets**: Iris, Wine, California Housing

### **Generated Artifacts Include:**
- Feature importance plots
- Confusion matrices
- Learning curves
- Residual plots
- Classification reports
- Model files
- Prediction CSVs

---

## 📋 Demo Checklist

**Before Starting:**
- [ ] MLflow server running on port 5000
- [ ] Main dashboard accessible
- [ ] All experiments visible
- [ ] Model registry populated

**During Demo:**
- [ ] Show main dashboard overview
- [ ] Navigate to Iris Classification experiment
- [ ] Demonstrate run comparison
- [ ] Show individual run details and artifacts
- [ ] Visit Model Registry
- [ ] Highlight different experiment types
- [ ] Emphasize key value propositions

**Demo Success Indicators:**
- [ ] Audience can see real experiments and data
- [ ] UI is responsive and professional
- [ ] All artifacts load correctly
- [ ] Model registry shows registered models
- [ ] Comparison tools work smoothly

---

**🎯 Total Demo Time**: 10-15 minutes for full walkthrough, 5-7 minutes for highlights only.

**🌐 Primary URL**: http://localhost:5000 - Start here and everything else is accessible!

*The MLflow environment is fully configured and ready for your workshop demonstration.*