# NASA Kepler Exoplanet Tournament

An advanced, multi-stage Machine Learning pipeline designed to classify cosmic observations from the NASA Kepler dataset. This project implements custom physical feature engineering and a tournament-style ensemble framework to accurately identify candidate, confirmed, and false-positive exoplanets.

## 🚀 Key Features

*   **Physical Feature Engineering:** Computes domain-specific metrics including log-transformed Signal-to-Noise Ratios (SNR), temperature-to-radius logs, and squared transit area ratios based on exoplanetary physics.
*   **Robust ML Pipelines:** Built using Scikit-Learn `Pipeline` architectures incorporating `SimpleImputer`, `StandardScaler`, and custom preprocessing steps to prevent data leakage.
*   **Tournament Ensemble Model:** Leverages a collection of `HistGradientBoostingClassifier` models optimized to handle multi-class target imbalances.
*   **Comprehensive Evaluation:** Generates multi-class Receiver Operating Characteristic (ROC) curves, Micro-average AUC metrics, and detailed confusion matrices to diagnose model performance.

## 📦 Requirements

*   Python 3.10+
*   Pandas
*   NumPy
*   Scikit-Learn
*   Matplotlib / Seaborn

## 🔧 Getting Started

1. **Clone the repository:**
```bash
   git clone [https://github.com/Devki164/nasa-kepler-exoplanet-tournament.git](https://github.com/Devki164/nasa-kepler-exoplanet-tournament.git)
   cd nasa-kepler-exoplanet-tournament
