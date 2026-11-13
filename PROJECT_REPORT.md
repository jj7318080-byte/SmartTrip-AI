# SmartTrip AI – Mini Project Report

## Aim
To develop a machine-learning-based travel planning website that predicts estimated trip expenses and recommends destinations according to a user's budget.

## Problem statement
Travellers may find it difficult to estimate the combined cost of transport, accommodation, food and local travel before planning a trip. This prototype provides an early budget estimate and suggests destinations with similar cost profiles.

## Objectives
- Build a web interface to collect trip preferences.
- Train a Random Forest Regressor to estimate total travel cost.
- Apply K-Nearest Neighbors to identify similar destination profiles.
- Display estimated expense components and save prediction history.

## Software and libraries
Python, Flask, HTML, CSS, JavaScript, SQLite, Pandas, NumPy, Scikit-learn and Joblib.

## Algorithms
**Random Forest Regressor:** An ensemble of decision trees used for numerical prediction. It combines the predictions of multiple trees to estimate trip cost.

**K-Nearest Neighbors:** A distance-based method. It scales destination profile features and retrieves nearby profiles using Euclidean distance.

## Dataset
The supplied dataset has 2,400 illustrative generated records and a 30-destination reference catalogue. The dataset is synthetic and must be identified as such in presentations. It is intended to demonstrate model training and application integration, not to claim verified market accuracy.

## Model evaluation
The trained Random Forest model is evaluated using an 80:20 train-test split. The generated metrics are stored in `ml_model/model_metrics.json` and shown on the website. MAE measures average absolute prediction error; R² indicates the proportion of target variance explained on the held-out sample. Since the dataset is synthetic, these metrics describe performance only on the generated data.

## Modules (connected user workflow)
1. Overview dashboard
2. Budget-based destination finder using KNN
3. Destination selection and trip details form
4. Travel cost prediction using Random Forest Regressor
5. Predicted total and expense breakdown
6. Trip history stored in SQLite
7. About and model information

**Workflow:** User enters a budget → receives destination suggestions → selects a destination → enters trip details → gets predicted cost and expense breakdown → prediction is saved in trip history.

## Result
The application runs locally and demonstrates a complete workflow: user input, trained ML inference, result visualization, KNN-based destination matching and SQLite history storage.

## Future enhancement
Use verified, dated fares and hotel prices; calculate route distances from origin; connect permitted transport/hotel APIs; add user authentication and cloud deployment.
