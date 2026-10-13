# AI-Based Indoor Air Quality Health Advisor

## 1. Project Overview

The **AI-Based Indoor Air Quality Health Advisor** is a machine learning-based application designed to analyze indoor air quality data and predict the air quality condition 30 minutes into the future.

Indoor air quality can be affected by carbon dioxide (CO₂), particulate matter (PM2.5 and PM10), volatile organic compounds (VOC), temperature, and humidity. Poor indoor air quality may cause discomfort and can contribute to health risks, particularly when people are exposed to pollutants for extended periods.

This project uses historical indoor air quality measurements to train machine learning models. The selected model predicts a future air quality category, while the application displays the prediction, confidence, feature explanations, and recommendations through an interactive Streamlit dashboard.

The system is intended for educational and analytical purposes and demonstrates how machine learning can support indoor environmental monitoring and early warnings.

## 2. Problem Statement

Indoor air quality is not always monitored continuously, and people may not recognize deteriorating conditions until the effects become noticeable.

A system that only displays current sensor measurements cannot indicate how conditions may change in the near future.

Therefore, this project aims to develop a machine learning-based solution that:
- Analyzes historical indoor air quality measurements.
- Learns patterns from the available dataset.
- Predicts the air quality category 30 minutes ahead.
- Provides early warnings when air quality is expected to deteriorate.
- Recommends actions that may help improve indoor air quality.
- Presents results through a simple and interactive dashboard.

## 3. Objectives

The main objectives of this project are:

1. To analyze an indoor air quality dataset using Python.
2. To preprocess the data and prepare it for machine learning.
3. To create features from current and historical sensor measurements.
4. To train and compare multiple machine learning classification models.
5. To select a model using validation performance.
6. To predict indoor air quality 30 minutes ahead.
7. To explain model predictions using feature-importance techniques.
8. To provide recommendations based on predicted air quality.
9. To visualize readings, predictions, and model evaluation results.
10. To develop a user-friendly web application using Streamlit.

## 4. Scope of the Project

The project focuses on software-based indoor air quality prediction using an existing dataset.

The application is designed to support three operating modes, depending on the implemented dashboard functionality:

- **Historical replay:** Examines a historical period and compares predictions with subsequent recorded outcomes.
- **Manual input:** Allows users to enter indoor air quality readings and obtain a prediction.
- **CSV upload:** Allows users to upload compatible data for analysis.

The project does not require physical sensor hardware to demonstrate its machine learning workflow.

## 5. Dataset Description

**Dataset filename:** `AQUAIR_1.csv`

**Dataset location:** `data/AQUAIR_1.csv`

The dataset contains timestamped indoor air quality measurements. It is used to identify patterns in the available readings and prepare data for future air quality prediction.

### Dataset attributes

| Attribute | Description |
|---|---|
| `timestamp(UTC)` | Timestamp associated with a measurement |
| `score` | Air quality score provided in the dataset |
| `temp` | Temperature measurement |
| `humid` | Relative humidity measurement |
| `co2` | Carbon dioxide concentration |
| `voc` | Volatile organic compound measurement |
| `pm25` | Fine particulate matter measurement |
| `pm10` | Coarser particulate matter measurement |

The dataset contains **16,533 rows and 8 columns**, with timestamps spanning October to December 2024, based on the initial dataset inspection.

### Data preprocessing

The preprocessing workflow includes:
- Loading the dataset using Pandas.
- Converting timestamps into an appropriate datetime format.
- Sorting measurements chronologically.
- Checking for missing values and invalid records.
- Handling unusable rows and gaps in the time series.
- Preparing chronological training, validation, and testing subsets.
- Creating historical features and future prediction targets where required.

Chronological splitting helps evaluate the model on later data rather than randomly mixing past and future observations.

## 6. Air Quality Classification

The application uses three air quality categories:

| Category | Interpretation |
|---|---|
| **GOOD** | Conditions are classified as good by the model or the project's classification rules. |
| **MODERATE** | Conditions indicate that air quality may require attention. |
| **POOR** | Conditions indicate potentially concerning indoor air quality. |

The predicted category represents the classification produced by the trained model. It should not be interpreted as a medical diagnosis or a certified safety assessment.

The exact category boundaries depend on the project's implemented labeling rules and should be documented alongside the training code.

## 7. Proposed System

The system follows this general workflow:

1. **Dataset collection:** Load indoor air quality measurements from the CSV dataset.
2. **Data preprocessing:** Clean and organize the data.
3. **Feature engineering:** Prepare current measurements and relevant historical features.
4. **Dataset splitting:** Divide the time-ordered data into training, validation, and testing subsets.
5. **Model training:** Train multiple classification algorithms.
6. **Model evaluation:** Compare models using suitable classification metrics.
7. **Model selection:** Select the model based on validation performance.
8. **Model saving:** Save the trained model and associated preprocessing information.
9. **Future prediction:** Estimate the air quality category 30 minutes ahead.
10. **Explanation:** Present available feature-contribution or importance information.
11. **Recommendation:** Generate advice based on the predicted category.
12. **Visualization:** Display the results in the Streamlit dashboard.

## 8. Machine Learning Algorithms

The training pipeline compares multiple machine learning algorithms. Depending on the final implementation, these may include:

- **Logistic Regression:** A classification algorithm used as a baseline.
- **Decision Tree:** Classifies observations through a sequence of decision rules.
- **Random Forest:** Combines predictions from multiple decision trees.
- **Extra Trees:** Uses randomized decision trees to learn classification patterns.
- **Histogram Gradient Boosting:** Builds an ensemble of trees through gradient boosting.
- **XGBoost:** An optimized gradient-boosting implementation.
- **Multilayer Perceptron (MLP):** A neural-network-based classification model.

The project selects a model based on the evaluation procedure implemented in the training pipeline. The final selected algorithm and its measured performance should be taken from the latest training output and saved comparison report.

## 9. Prediction Horizon

The system is designed to predict the indoor air quality category **30 minutes ahead**.

The prediction process uses the available current and historical information to estimate the future category. Where the dataset has regular five-minute readings, a 30-minute horizon corresponds to six five-minute intervals.

The historical feature-generation logic must respect chronological order to avoid using future information when predicting an earlier time.

The model estimates a future category; it does not guarantee that the actual indoor conditions will match the prediction.

## 10. Model Evaluation

The models are evaluated using classification metrics such as:

- **Accuracy:** The proportion of correctly classified observations.
- **Precision:** The proportion of predicted class instances that are correct.
- **Recall:** The proportion of actual class instances correctly identified.
- **F1-score:** A combined measure of precision and recall.
- **Macro-F1:** The average F1-score across classes, giving each class equal weight.
- **Balanced accuracy:** The average recall across classes.

Accuracy alone may be misleading if GOOD, MODERATE, and POOR observations are unevenly distributed. Macro-F1 and per-class recall are particularly useful for examining whether the model identifies less common poor-air-quality cases.

The held-out test set should be used for final evaluation, rather than repeatedly adjusting the model based on test performance.

Actual metrics should be reported from the latest generated model comparison and evaluation files.

## 11. Explainability

The project includes a prediction-explanation module intended to help users understand which features contribute to model predictions.

Where supported by the selected model and implementation, the system can display feature contributions or feature-importance information.

These explanations describe the model's behavior. They do not establish that a particular feature directly caused a change in indoor air quality.

## 12. Recommendation System

The recommendation module provides guidance according to the predicted air quality category.

Examples of general recommendations include:

- **GOOD:** Continue normal indoor ventilation and monitoring.
- **MODERATE:** Consider improving ventilation and monitoring changes in indoor air quality.
- **POOR:** Improve ventilation where safe, investigate possible pollution sources, and consider leaving the affected space if conditions are concerning.

Recommendations depend on the rules implemented in `src/recommendations.py`. They should be treated as general guidance rather than a replacement for certified monitoring, professional advice, or emergency procedures.

## 13. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset processing and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| XGBoost | Gradient-boosting model |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web dashboard |
| SHAP | Model explanation support |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Pytest | Automated testing |

## 14. Project Structure

```text
AIML_FA2_IAQ_Project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── AQUAIR_1.csv
│
├── models/
│   └── best_iaq_30min_model.pkl
│
├── src/
│   ├── common.py
│   ├── experiments.py
│   ├── explain.py
│   ├── modeling.py
│   ├── recommendations.py
│   └── train_models.py
│
├── reports/
│   └── Generated model evaluation tables and reports
│
├── report/
│   └── Project documentation, if maintained here
│
├── tests/
│   └── Automated test files, if present
│
└── .venv/
    └── Local Python virtual environment
```

The `.venv/` directory is created locally and generally should not be included when sharing the project. Some folders or output files may only appear after training, running experiments, or generating reports.

## 15. Installation and Setup

### Prerequisites

Install Python and ensure that `python` and `pip` are available in PowerShell.

Open PowerShell and navigate to the project directory:

```powershell
cd "$HOME\Desktop\AIML\AIML_FA-2\AIML_FA2_IAQ_Project"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

These setup steps normally need to be performed once per environment. Activate the environment again when opening a new PowerShell terminal.

## 16. Training the Machine Learning Models

Before training, check that the dataset exists at `data/AQUAIR_1.csv` and that the required source files are present.

Run:

```powershell
python src\train_models.py
```

The training pipeline is intended to:
1. Load and prepare the dataset.
2. Generate the required features and target labels.
3. Create chronological data splits.
4. Train and compare candidate models.
5. Evaluate model performance.
6. Select a model according to the implemented selection metric.
7. Save the selected model and evaluation outputs.

The precise files generated depend on the current implementation of `src/train_models.py`.

If a compatible trained model already exists, retraining is generally unnecessary for ordinary dashboard use unless the model needs to be updated or regenerated.

## 17. Running the Streamlit Dashboard

After dependencies are installed and the trained model is available, run:

```powershell
streamlit run app.py
```

Streamlit displays a local URL in the terminal. Open that URL in your browser to access the dashboard.

The dashboard is intended to display the supported input mode, current measurements, predicted air quality category 30 minutes ahead, prediction confidence, available explanations, and recommendations.

The precise dashboard features depend on the current implementation of `app.py`.

To stop the application, press `Ctrl+C` in the terminal running Streamlit.

## 18. Additional Commands

### Generate experiment reports

If `src/experiments.py` and its required inputs are available:

```powershell
python src\experiments.py
```

Use this command when you need additional model comparisons, plots, or experiment outputs for your academic report.

### Run automated tests

If the project contains tests:

```powershell
python -m pytest tests -v
```

This command executes the available tests and reports whether they pass or fail.

These commands are optional for ordinary dashboard use.

## 19. Expected Output

When the project runs successfully, it should provide:

- An interactive indoor air quality dashboard.
- Analysis of compatible indoor air quality readings.
- A machine learning prediction for the air quality category 30 minutes ahead.
- Prediction confidence, where implemented.
- Historical or uploaded-data analysis, where supported.
- Feature explanations, where supported.
- Category-based recommendations.
- Model comparison and evaluation outputs after training.

The actual predictions and model metrics depend on the dataset, training process, and saved model.

## 20. Limitations

- The prediction quality depends on the quality and coverage of the training dataset.
- Rare POOR observations may be harder for the model to classify reliably.
- Historical relationships may not represent every building or indoor environment.
- Predictions can be inaccurate when conditions change unexpectedly.
- Model explanations do not prove causal relationships.
- The project does not replace certified indoor air quality monitoring equipment.
- Manual inputs and uploaded files must follow the format expected by the application.
- A prediction 30 minutes ahead is an estimate, not a guarantee.

## 21. Future Enhancements

Possible future improvements include:

- Adding more diverse indoor air quality datasets.
- Improving detection of rare poor-air-quality conditions.
- Comparing additional time-series forecasting methods.
- Improving dashboard visualizations and historical trend analysis.
- Adding more comprehensive automated tests.
- Supporting deployment to a cloud platform.
- Evaluating model performance on independent indoor environments.

## 22. Conclusion

The AI-Based Indoor Air Quality Health Advisor demonstrates how machine learning can be applied to indoor environmental data to estimate future air quality conditions.

By combining data preprocessing, historical feature engineering, model comparison, future classification, explainability, recommendations, and an interactive Streamlit interface, the project provides an educational example of predictive indoor air quality analysis.

The system is intended to support awareness and early intervention, while recognizing the limitations of data-driven predictions and the importance of reliable air quality measurements.

## 23. Author and Academic Information

**Project Title:** AI-Based Indoor Air Quality Health Advisor

**Department:** Information Technology

**Institution:** PCCoE Pune

**Academic Use:** AIML project / academic submission