# Bias–Variance Tradeoff in Machine Learning

This project explores the **bias–variance tradeoff** on the California Housing dataset.  
It demonstrates how **model complexity, feature engineering, and regularization** affect training vs test error.

---

## 🔑 Key Experiments

- **Complexity Sweep**  
  Polynomial degree ↑ → training error ↓, test error shows U-shaped curve (bias ↓, variance ↑).

- **Feature Engineering**  
  Domain-informed features (rooms/person, bedroom ratio, coastal flag) outperform raw polynomial expansions.

- **Regularization (Ridge & Lasso)**  
  Prevents variance explosion at high degrees; optimal α balances bias and variance.

- **Learning Curves**  
  More data → test error ↓ as variance shrinks.

- **Bootstrap Decomposition**  
  Bias² ↓ and variance ↑ with complexity; their sum tracks test MSE.

---

## 📊 Sample Results

| Experiment              | Plot |
|--------------------------|------|
| Complexity Sweep (OLS)   | ![Complexity](docs/figures/complexity_sweep.png) |
| Ridge Regularization     | ![Ridge](docs/figures/regularization_sweep.png) |
| Bias–Variance Bootstrap  | ![Bootstrap](docs/figures/bias_variance_bootstrap.png) |

---

## ⚙️ How to Run

```bash
git clone https://github.com/<your-username>/bias-variance-california.git
cd bias-variance-california
pip install -r requirements.txt
python Bias_Variance_Spyder.py
