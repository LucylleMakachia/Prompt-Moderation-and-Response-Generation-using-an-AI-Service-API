# Prompt-Moderation-and-Response-Generation-using-an-AI-Service-API
# 🏠 Linear Regression Model – Predicting Home Prices

## 📘 Overview
This project demonstrates a simple **Linear Regression** model using **Python** and **Scikit-learn** to predict **house prices** based on **area (square feet)**.  
It is part of a data science learning assignment designed to build hands-on experience in:
- Exploring and cleaning a real-world dataset  
- Training and testing a machine learning model  
- Evaluating model performance using standard metrics  
- Visualizing predictions and regression lines  

---

## 🧠 Project Objectives
The key goal is to predict the price of a house given its area, using a simple linear regression model.

You will learn to:
1. Load and explore a dataset with `pandas`
2. Visualize data relationships with `matplotlib` and `seaborn`
3. Train a regression model using `scikit-learn`
4. Evaluate it using **MAE**, **MSE**, **RMSE**, and **R²**
5. Plot the regression line and interpret the results

---

## 📂 Dataset
**File:** `homeprices.csv`  

| Column | Description |
|:-------|:-------------|
| `area` | Area of the house in square feet |
| `price` | Selling price of the house in USD |

#### Example Data
| area | price |
|------|--------|
| 2600 | 550000 |
| 3000 | 565000 |
| 3200 | 610000 |
| 3600 | 680000 |

---

## 🧩 Project Structure
```
LinearRegression_HomePrices/
│
├── homeprices.csv              # Dataset
├── Linear_Regression_Model.ipynb  # Jupyter/Colab Notebook
├── README.md                   # Project documentation (this file)
└── model.pkl                   # Saved regression model (optional)
```

---

## ⚙️ Technologies Used
- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter / Google Colab  

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/<yourusername>/LinearRegression_HomePrices.git
   cd LinearRegression_HomePrices
   ```

2. **Open in Google Colab**
   - Upload `homeprices.csv`
   - Open `Linear_Regression_Model.ipynb`
   - Run all cells sequentially

3. **Install required libraries**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

4. **Run the notebook**
   The notebook will:
   - Load and explore the dataset  
   - Train and evaluate a Linear Regression model  
   - Plot the regression line  
   - Predict new house prices

---

## 📊 Model Evaluation Metrics
| Metric | Meaning | Goal |
|:--------|:---------|:------|
| **MAE** | Mean Absolute Error | Lower = better |
| **MSE** | Mean Squared Error | Lower = better |
| **RMSE** | Root Mean Squared Error | Lower = better |
| **R² Score** | Variance explained by the model | Closer to 1 = better |

---

## 📈 Visualization
The scatter plot and regression line show how predicted prices align with actual prices.

```python
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend()
plt.show()
```

---

## 💡 Key Insights
- There is a **positive correlation** between area and price — larger houses tend to cost more.  
- The **Linear Regression model** performs well for this simple one-variable dataset.  
- The project illustrates the **complete ML workflow** from data exploration to prediction.

---

## 🧾 Sample Prediction
```python
new_area = 3000
predicted_price = model.predict([[new_area]])
print(f"Predicted price for {new_area} sq ft = ${predicted_price[0]:,.2f}")
```

---

## 🧠 What I Learned
- How to build, evaluate, and interpret a simple linear regression model  
- How to visualize data relationships and model predictions  
- The importance of proper data splitting and performance metrics  
- How to structure and document a machine learning project for sharing  

---

## 🔗 Project Links
- 📄 **Google Colab Notebook:** [[https://colab.research.google.com/drive/1yiczgSS8iW2l0m3yZ3DM4oqh3F8T63V0?usp=sharing]]
- 🧠 **Portfolio Page:** [Add your portfolio link if available]
- 💾 **Dataset:** `homeprices.csv`

---

## ✍️ Author
**Lucylle Makachia**  
Data Science & GIS Analyst | Web Developer | Climate Tech Innovator  
📍 Nairobi, Kenya  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/lucylle-makachia)

