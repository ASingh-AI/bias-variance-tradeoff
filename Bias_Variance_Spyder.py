# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 08:10:10 2025

@author: as972
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

### SET FOLDERS RELATIVE TO CURRENT PYTHON SCRIPT ###

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR/"plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[i] Results folder: {RESULTS_DIR}")
print(f"[i] Plots folder:   {PLOTS_DIR}")

### EXPERIMENTS ###

### 1 -> Load data (as pandas DataFrame) + (optional) engineered features

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    eps = 1e-6
    X["Rooms_per_person"] = X["AveRooms"] / np.maximum(X["AveOccup"], eps)
    X["Bedroom_ratio"] = X["AveBedrms"] / np.maximum(X["AveRooms"], eps)
    X["Close_to_beach_flag"] = ((X["Longitude"] > -122) & (X["Latitude"].between(33,38))).astype(int)
    return X

def load_data(use_engineered: bool=False):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    y = df["MedHouseVal"]
    X_raw = df.drop(columns=["MedHouseVal"])
    X = add_engineered_features(X_raw) if use_engineered else X_raw
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test


### 2 -> Model Builders + Evaluators

def model_linear (degree: int) -> Pipeline:
    return make_pipeline(
        PolynomialFeatures(degree = degree, include_bias = False),
        StandardScaler(with_mean=False),
        LinearRegression()
    )

def model_ridge (degree: int, alpha: float=1.0) -> Pipeline:
    return make_pipeline(
        PolynomialFeatures(degree = degree, include_bias = False),
        StandardScaler(with_mean=False),
        Ridge(alpha=alpha, random_state = 42)
    )
        
def model_lasso (degree: int, alpha: float = 0.1) -> Pipeline:
    # L1 regularization can zero-out coefficients (feature selection).
    # Use a higher max_iter and a small tol to avoid convergence warnings.
    return make_pipeline(
        PolynomialFeatures(degree = degree, include_bias = False),
        StandardScaler(with_mean=False),
        Lasso(alpha=alpha, random_state = 42, max_iter = 50000, tol=1e-4)
    )

def eval_model(model:Pipeline, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    training_error = mean_squared_error(y_train, model.predict(X_train))
    test_error = mean_squared_error(y_test, model.predict(X_test))
    return training_error, test_error
    
### 3 -> Complexity sweep → Train ↓, Test U-shaped (bias ↓, variance ↑)

def sweep_complexity(X_train, X_test, y_train, y_test, degrees, modelname, label, outfile):
    rows = []
    for d in degrees:
        model = modelname(d)
        training_error, test_error = eval_model(model, X_train, X_test, y_train, y_test)
        rows.append(dict(
        kind=label, 
        degree =d, 
        train_mse = training_error, 
        test_mse=test_error
    ))
        
    df = pd.DataFrame(rows).sort_values("degree")

    plt.figure(figsize=(7,4))
    plt.plot(df["degree"], df["train_mse"], marker="o", label=f"Train MSE ({label})")
    plt.plot(df["degree"], df["test_mse"],  marker="o", label=f"Test MSE ({label})")
    plt.xlabel("Polynomial Degree"); plt.ylabel("MSE"); plt.title("Complexity Sweep")
    plt.legend() 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savepath = PLOTS_DIR / outfile
    plt.savefig(savepath, dpi=160)
    plt.show()
    
    return df


### 4 -> Features → Better features shift the curve down -> Engineered features are always better than Polynomial Features

def compare_features(degrees, modelname, outfile):
    rows = []
    
    for use_eng in [False, True]:
        X_train, X_test, y_train, y_test = load_data(use_engineered=use_eng)
        tag = "eng" if use_eng else "raw"
        for d in degrees:
            model = modelname(d)
            training_error, test_error = eval_model(model, X_train, X_test, y_train, y_test)
            rows.append(dict(
            features=tag, 
            degree =d, 
            train_mse = training_error, 
            test_mse=test_error
    ))
        
    df = pd.DataFrame(rows).sort_values(["features","degree"])

    plt.figure(figsize=(7,4))
    for tag, dfi in df.groupby("features"):
       plt.plot(dfi["degree"], dfi["test_mse"], marker="o", label=f"Test MSE ({tag}")
    plt.xlabel("Polynomial Degree"); 
    plt.ylabel("MSE")
    plt.title("Effect of Engineered Features")
    plt.legend() 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savepath = PLOTS_DIR / outfile
    plt.savefig(savepath, dpi=160)
    plt.show()
    
    return df


### 5 -> Regularization → Ridge smooths out variance explosion

def sweep_regularization(X_train, X_test, y_train, y_test, degree, alphas, outfile):
    rows = []
    for a in alphas:
        model = model_ridge(degree=degree, alpha=a)
        training_error, test_error = eval_model(model, X_train, X_test, y_train, y_test)
        rows.append(dict(
            alpha=a, 
            train_mse = training_error, 
            test_mse=test_error,
        ))    
        
    df = pd.DataFrame(rows)

    plt.figure(figsize=(7,4))
    plt.plot(df["alpha"], df["train_mse"], marker="o", label="Train MSE")
    plt.plot(df["alpha"], df["test_mse"], marker="o", label="Test MSE")
    plt.xscale("log")
    plt.xlabel("Ridge Alpha (log scale)")
    plt.ylabel("MSE")
    plt.title("Regularization Sweep")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    savepath = PLOTS_DIR / outfile
    plt.savefig(savepath, dpi=160)
    plt.show()
            
    return df
    
### 6 -> Learning curve → More data shrinks variance → test MSE ↓

def learning_curve(X, y, degree, alpha, train_sizes, outfile):
    
    rows = []
    for frac in train_sizes:
        X_train, _, y_train, _, = train_test_split(X, y, train_size=frac, random_state=42)
        X_test, y_test = X, y # Evaluate on full dataset
        model = model_ridge(degree=degree, alpha=alpha)
        training_error, test_error = eval_model(model, X_train, X_test, y_train, y_test)
        rows.append(dict(
            train_frac=frac, 
            train_mse = training_error, 
            test_mse=test_error,
        ))    
        
    df = pd.DataFrame(rows)
            
    plt.figure(figsize=(7,4))
    plt.plot(df["train_frac"], df["train_mse"], marker="o", label="Train MSE")
    plt.plot(df["train_frac"], df["test_mse"], marker="o", label="Test MSE")
    plt.xlabel("Fraction of Training Data")
    plt.ylabel("MSE")
    plt.title("Learning Curve")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    savepath = PLOTS_DIR / outfile
    plt.savefig(savepath, dpi=160)
    plt.show()
            
    return df
    
### 7 ->  Bootstrap → Bias² ↓, variance ↑, sum ≈ test MSE

def bootstrap_bias_variance(X_train, X_test, y_train, y_test, degrees, n_boot=50):
    ### """Estimate bias^2 and variance via bootstrap resampling.""" ###
    results = []
    for d in degrees:
        preds=[]
        for b in range(n_boot):
            # resample withreplacement
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=len(X_train),replace=True)
            Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
            model = model_linear(d)
            model.fit(Xb, yb)
            preds.append(model.predict(X_test))
        preds = np.array(preds) ## shape - (n_boot, n_samples)
        
        avg_pred = preds.mean(axis=0)
        bias2 = np.mean((avg_pred - y_test.values) ** 2)
        var = np.mean(preds.var(axis=0))
        mse = bias2 + var
        results.append(dict(
                degree=d, 
                bias2=bias2, 
                variance=var,
                mse=mse
            ))
        print(f"Degree={d:2d} | Bias²={bias2:.4f} | Var={var:.4f} | MSE={mse:.4f}")
            
    df = pd.DataFrame(results)
     
     
    plt.figure(figsize=(7, 4))
    plt.plot(df["degree"], df["bias2"], marker="o", label="Bias²")
    plt.plot(df["degree"], df["variance"], marker="o", label="Variance")
    plt.plot(df["degree"], df["mse"], marker="o", label="Bias²+Variance")
    plt.xlabel("Polynomial Degree"); plt.ylabel("Error"); plt.title("Bias–Variance Decomposition (Bootstrap)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_bias_variance_bootstrap.png", dpi=160)
    plt.show()
    return df

# ------------------------------------

# Main - Executing all options 

# ------------------------------------

if __name__ == "__main__":
    degrees = [1,2,3,5]
    ridge_degrees = [1,2,3,5,8,10]
    # Load Baseline
    X_train, X_test, y_train, y_test = load_data(use_engineered=False)
    X_full, y_full = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
                               
    # Complexity Sweeps
    sweep_complexity(X_train, X_test, y_train, y_test, degrees, model_linear, "LinearModel-OLS", "01_Complexity_sweep_OLS.png")   
    sweep_complexity(X_train, X_test, y_train, y_test, ridge_degrees, lambda d: model_ridge(d, alpha=1.0), "Ridge Aplha=1.0", "02_Complexity_sweep_Ridge.png") 


    # 3) Regularization sweep
    sweep_regularization(X_train, X_test, y_train, y_test, degree=8, alphas=[0.01, 0.1, 1.0, 10, 100], outfile="04_regularization_sweep.png")

    # 4) Learning curve
    learning_curve(X_full, y_full, degree=5, alpha=1.0, train_sizes=[0.05, 0.1, 0.2, 0.5, 0.99], outfile="05_learning_curve.png")

    # 5) Bootstrap bias–variance decomposition
    df_boot = bootstrap_bias_variance(X_train, X_test, y_train, y_test, degrees)
    print("\nBootstrap bias-variance decomposition results:")
    print(df_boot.to_string(index=False))






                        