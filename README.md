# Clustering and Fitting Assignment

**Course:** Applied Data Science 1 
**Instructor:** Dr. William Cooper 
**Student:** Mohammed Abdul Jilani 
**Student Number:** 24168848
**Institution:** University of Hertfordshire 
**Date:** November 2025

---

## 📋 Description

This project performs comprehensive customer segmentation analysis using k-means clustering and polynomial curve fitting on the Mall Customer Segmentation dataset. The analysis identifies distinct customer behavioral groups and models the relationship between annual income and spending patterns.

---

## 📊 Dataset

**Source:** [Mall Customer Segmentation - Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

**File:** `data.csv`

**Contents:** 200 customer records with Age, Gender, Annual Income, and Spending Score

---

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/mohammedabduljilani07/clustering-fitting-assignment.git
cd clustering-fitting-assignment

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage
```bash
python clustering_and_fitting.py
```

**Output:** 6 PNG visualization files

---

## 📈 Key Results

- **Optimal Clusters:** k=5 (Silhouette Score: 0.55)
- **Customer Segments:** 
  - High-Value (39 customers)
  - Affluent Cautious (23)
  - Budget-Conscious (35)
  - Aspirational Spenders (38)
  - Middle-Income Moderates (65)
- **Income-Spending Correlation:** r=0.009 (very weak)
- **Fitted Model:** y = 0.0007x² - 0.1072x + 54.87

---

## 📁 Project Structure
```
clustering-fitting-assignment/
├── README.md
├── clustering_and_fitting.py
├── data.csv
├── requirements.txt
├── relational_plot.png
├── categorical_plot.png
├── statistical_plot.png
├── elbow_plot.png
├── clustering.png
└── fitting.png
```

---

## 🎯 Methods

- **Statistical Analysis:** Mean, Std Dev, Skewness, Kurtosis
- **Clustering:** K-means with RobustScaler normalization
- **Optimization:** Elbow method & Silhouette analysis
- **Curve Fitting:** Polynomial regression with scipy
- **Visualization:** 6 professional plots with matplotlib/seaborn

---

## 📝 Report

See `clustering_fitting_report.pdf` for complete 3-page analysis report.

---

## 👤 Author

Mohammed Abdul Jilani  
University of Hertfordshire  
Applied Data Science 1 - November 2025

---

## 📄 License

Submitted as coursework. All rights reserved.
