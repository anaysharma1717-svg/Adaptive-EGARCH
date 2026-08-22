# Adaptive EGARCH: SPY Volatility Forecasting

This repository contains a quantitative research pipeline and a deployment-ready trading strategy for forecasting SPY Realized Volatility using an adaptive EGARCH framework.

## Overview

The core objective of this project is to accurately forecast the 21-day Garman-Klass Realized Volatility (GK-RV) of the SPY ETF to drive an options straddle arbitrage algorithm. The research pipeline evaluates multiple econometric models, focusing heavily on capturing the leverage effect (via EGARCH) and long-memory dependencies (via Heterogeneous Autoregressive components).

### Key Features
*   **Strict Walk-Forward Evaluation**: All models are evaluated out-of-sample with strict expanding windows to ensure zero look-ahead bias.
*   **Garman-Klass Realized Volatility**: Uses High-Low-Open-Close data to construct a highly efficient, unbiased daily variance estimator.
*   **Extended Model Zoo**: Evaluates standard EGARCH alongside Augmented HAR models (Log-HAR, Asymmetric HAR, HAR+Returns) with Mincer-Zarnowitz efficiency and Fair-Shiller forecast encompassing tests.
*   **QuantConnect Deployment Integration**: Automatically generates properly scaled, annualized forecast JSONs and seamlessly injects them into a QuantConnect algorithmic trading script for cloud deployment.

## Repository Structure

*   `run_pipeline.py`: The main entry point. Fetches data, runs the walk-forward evaluation for the core models, and outputs the OOS volatility forecasts.
*   `inject_forecasts.py`: A utility that reads the generated forecasts, applies base64 compression, and directly injects them into the QuantConnect `main.py` file for deployment.
*   `quantconnect_backtest/`: Contains the deployment-ready Lean CLI algorithm (`main.py`) which executes a delta-hedged 21-day ATM options straddle based on the forecasts.
*   `research/`: An isolated environment containing the extended model zoo, diagnostic plotting scripts, experimental results, and historical project logs.

## Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Research Pipeline**:
    Execute the pipeline to generate 21-day annualized out-of-sample forecasts over a rolling 6-month window (configurable).
    ```bash
    python run_pipeline.py --start 2016-01-01 --test-months 6
    ```

3.  **Prepare for Deployment**:
    Inject the generated forecasts into the QuantConnect algorithm.
    ```bash
    python inject_forecasts.py
    ```

4.  **Run Backtest**:
    If you have QuantConnect Lean CLI installed, you can launch the backtest using the provided PowerShell script.
    ```powershell
    ./run_lean_cloud.ps1
    ```

## Research Findings

Extensive testing across 3900+ trading days demonstrated that while standard EGARCH(1,1) effectively captures crisis-period shocks (the leverage effect), its standalone signal is too noisy for optimal out-of-sample forecasting compared to simpler HAR-RV models. The best-performing specifications combine Heterogeneous Autoregressive (HAR) features with asymmetric (negative semi-variance) components, providing superior accuracy during calm regimes while maintaining responsiveness during drawdowns.
