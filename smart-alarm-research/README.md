# AI-Based Smart Alarm Using Heart Rate and Sleep Data

This project explores whether heart rate, sleep, and lifestyle patterns can help predict sleep quality and support future smart alarm optimization.

## Research Question

Can heart rate and lifestyle variables such as physical activity, stress level, sleep duration, and daily habits be used to predict sleep quality or related sleep outcomes?

## Hypothesis

People with healthier sleep duration, lower stress levels, and more balanced lifestyle patterns will tend to show better sleep quality. Heart rate and activity-related signals may provide useful clues for identifying better wake-up timing.

## Dataset Description

The project is built to work with CSV datasets such as:

- Sleep Health and Lifestyle Dataset
- Sleep Efficiency Dataset
- Smartwatch or wearable sleep tracking datasets

Place a CSV file in `data/raw/`, then update the `DATA_PATH` variable in `notebooks/01_eda_sleep_analysis.ipynb`.

## Technologies Used

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook
- VS Code

## Methods

The project includes:

- Data loading with clear error messages
- Column name cleaning
- Missing value handling
- Categorical variable encoding
- Exploratory data analysis
- Distribution and correlation visualizations
- Baseline machine learning models
- Model saving for future experiments

## Machine Learning Approach

The notebook automatically searches for a target column in this order:

1. Sleep Quality
2. Sleep Disorder
3. Sleep Efficiency

If the target is categorical, the project trains classification models:

- Logistic Regression
- Random Forest Classifier

If the target is continuous, the project trains regression models:

- Linear Regression
- Random Forest Regressor

## Results

The project achieved strong baseline machine learning performance for sleep quality prediction.

### Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | 94.6% |
| Random Forest | 97.3% |

### Key Findings

- Sleep duration showed a strong positive correlation with sleep quality.
- Stress level showed a strong negative correlation with sleep quality.
- Higher heart rate was associated with lower sleep quality.
- Random Forest achieved the best predictive performance.

## How to Run

1. Open the project folder in VS Code.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your CSV dataset to:

```text
data/raw/
```

5. Open and run:

```text
notebooks/01_eda_sleep_analysis.ipynb
```

Graphs are saved to `outputs/graphs/`.

Trained models are saved to `outputs/models/`.

## Expected Outcomes

This project should help identify which lifestyle and wearable variables are most related to sleep quality. The final result can support a future smart alarm system that wakes users during better sleep conditions instead of using a fixed alarm time only.

## Portfolio Value

This is structured as a research-style data science project rather than a single notebook. It shows data cleaning, visualization, baseline modeling, and written scientific communication.

## Future Work

Potential future improvements include:

- Real-time smartwatch integration
- Streamlit dashboard for interactive predictions
- Time-series sleep analysis
- Deep learning approaches
- Personalized wake-up optimization
- Integration with wearable APIs