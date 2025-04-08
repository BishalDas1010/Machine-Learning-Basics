# 📊 Linear Regression from Scratch in Python

This project implements **Linear Regression from scratch** (without using scikit-learn's regression model) to predict placement packages based on CGPA. It's a simple demonstration of the mathematics behind regression and data visualization using `matplotlib`.

---

## 📁 Dataset
The dataset used is `placement (2).csv` which contains:
- `cgpa`: Student's CGPA
- `package`: Placement package received (in LPA)

---

## 📌 Features
- Manual implementation of Linear Regression (no use of `LinearRegression` from sklearn)
- Calculates:
  - **Slope (m)**
  - **Intercept (b)**
- Predicts values using the regression equation
- Plots data points and the regression line for visualization

---

## 🧐 Math Behind the Model
We use the Least Squares Method:

\[
m = \frac{\sum{(x - \bar{x})(y - \bar{y})}}{\sum{(x - \bar{x})^2}} \quad , \quad b = \bar{y} - m \bar{x}
\]

Where:
- \( m \): slope of the line
- \( b \): y-intercept

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

2. **Install the required libraries**
```bash
pip install pandas matplotlib scikit-learn
```

3. **Run the script**
```bash
python linear_regression.py
```

---

## 📊 Output
- Green dots = actual data points
- Red line = regression prediction
- Console prints slope and intercept

---

## 👨‍💼 Author
- **Bishal Das**  
  B.tech Student | Aspiring Software & AI Developer
  
GitHub: [@BishalDas1010](https://github.com/BishalDas1010)

---

## 📄 License
This project is for learning purposes. Free to use and modify.

