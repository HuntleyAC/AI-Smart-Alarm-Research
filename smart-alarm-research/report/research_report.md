# AI-Based Smart Alarm Using Heart Rate and Sleep Data

## Abstract

This project investigates whether heart rate, sleep, and lifestyle data can be used to predict sleep quality and support smart alarm optimization. Public sleep and wearable-related datasets were analyzed using exploratory data analysis and machine learning methods.

The study identified strong relationships between stress level, sleep duration, heart rate, and sleep quality. Baseline machine learning models achieved high predictive performance, with Random Forest reaching approximately 97% accuracy.

The findings suggest that wearable and lifestyle-related metrics may support future AI-based smart alarm systems and personalized sleep optimization.

## Introduction

Traditional alarms wake users at a fixed time, regardless of sleep stage, stress, or physiological state. Smart alarms may improve wake-up experience by considering signals such as sleep duration, heart rate, physical activity, and stress level.

## Research Question

Can heart rate and lifestyle patterns help predict sleep quality or related sleep outcomes such as sleep disorder status or sleep efficiency?

## Dataset

The project is designed for CSV datasets related to sleep health, lifestyle, sleep efficiency, or smartwatch tracking. Potential variables include sleep duration, heart rate, physical activity, stress level, age, occupation, BMI category, and sleep quality.

## Methodology

The analysis follows these steps:

1. Load a CSV dataset.
2. Clean column names.
3. Handle missing values.
4. Explore important sleep and lifestyle variables.
5. Visualize distributions and correlations.
6. Select a prediction target.
7. Train baseline machine learning models.
8. Save graphs and trained models.

## Exploratory Data Analysis

The notebook checks the first rows, dataset structure, missing values, numeric summaries, distributions, and correlations. These steps help identify whether variables such as heart rate, stress, physical activity, and sleep duration appear related to sleep outcomes.

## Machine Learning Approach

The project predicts Sleep Quality when available. If Sleep Quality is missing, it attempts to predict Sleep Disorder or Sleep Efficiency. Classification targets use Logistic Regression and Random Forest. Continuous targets use Linear Regression and Random Forest regression.

## Results

The machine learning models achieved strong baseline performance for predicting sleep quality.

### Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | 94.6% |
| Random Forest | 97.3% |

The Random Forest model achieved the best performance and was selected as the final baseline model.

### Correlation Findings

Several important relationships were identified during the analysis:

- Sleep duration showed a strong positive correlation with sleep quality (0.88).
- Stress level showed a strong negative correlation with sleep quality (-0.90).
- Heart rate negatively correlated with sleep quality (-0.66).
- Physical activity level strongly correlated with daily steps (0.77).

These findings suggest that wearable and lifestyle metrics may contain meaningful predictive signals for smart alarm systems and sleep optimization.

## Limitations

Possible limitations include small dataset size, self-reported sleep quality, missing smartwatch signals, limited heart rate detail, and lack of real sleep stage data. Public datasets may not fully represent individual sleep patterns.

## Conclusion

This research demonstrates that sleep quality can be predicted with high accuracy using wearable and lifestyle-related variables such as stress level, heart rate, sleep duration, and physical activity.

The Random Forest model achieved approximately 97% accuracy, indicating that machine learning methods may be useful for future AI-based smart alarm systems.

The results support the idea that wearable device data can help improve wake-up timing and personalized sleep optimization.

## Future Work

Future improvements could include real smartwatch time-series data, sleep stage detection, personalized models, alarm-window optimization, and mobile app integration.
