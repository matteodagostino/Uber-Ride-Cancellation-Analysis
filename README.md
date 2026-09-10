# Uber Ride Cancellation Analysis

An academic data-science project analyzing **150,000 ride-booking records** to understand cancellation behavior and compare how classification models perform under different feature-availability assumptions.

The project was completed as a two-person course project. I developed the **complete Python notebook and technical analysis**, while my classmate collaborated on the written reporting and presentation of the work.

## Project Goal

The analysis focused on two related questions:

1. How accurately can ride cancellations be classified using the information available in the full dataset?
2. How does model performance change when the analysis is restricted to information that would be available before the ride outcome is known?

The second question became especially important after exploratory analysis showed that several missing values and post-ride fields were strongly connected to whether a ride had been cancelled.

## Dataset

The notebook uses a ride-booking dataset containing **150,000 observations and 21 original variables**. Features include booking status, vehicle type, pickup and drop locations, average vehicle/customer arrival times, cancellation indicators and reasons, booking value, ride distance, driver and customer ratings, and payment method.

The source CSV is not redistributed in this repository. The notebook expects the file `ncr_ride_bookings.csv` in its working directory.

## My Contribution

I developed the full Python analysis notebook, including:

- Initial data inspection and exploratory data analysis
- Missing-value investigation and data cleaning
- Construction of a binary `Cancelled` target variable from customer- and driver-cancellation indicators
- Categorical encoding and feature preparation
- Feature scaling for models sensitive to scale
- Stratified train/test splitting to preserve the cancellation class balance
- Training and comparison of multiple classification algorithms
- Evaluation using accuracy, precision, recall, F1 score, classification reports, and confusion matrices
- Investigation of feature availability and potential outcome leakage
- Construction of a second, pre-booking feature set and comparison against the full-data results
- Interpretation of the differences between the two modeling scenarios

## Analysis Workflow

### 1. Data Inspection and Cleaning

The notebook begins by examining data types, missing values, duplicate records, categorical distributions, and the overall structure of the dataset.

A binary `Cancelled` target is then created so that a ride is classified as cancelled when either the customer or the driver cancelled it.

### 2. Missingness as Information

Rather than treating every missing value as a random data-quality problem, the analysis checks whether missingness is related to the ride outcome. Several variables that describe completed rides become unavailable when a ride is cancelled—for example, ride distance and ratings.

This observation motivated a distinction between information that exists **after the ride outcome** and information that could plausibly be available **before the outcome is known**.

### 3. Preprocessing

The analysis prepares numeric and categorical predictors, encodes categorical variables, and scales the feature matrix where appropriate. The data is divided into an **80/20 stratified train/test split**, maintaining the approximately 75% non-cancelled / 25% cancelled class distribution.

### 4. Model Comparison

The notebook compares several classifiers:

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Random Forest
- SGD Classifier
- XGBoost
- Gradient Boosting

Performance is evaluated using accuracy, precision, recall, F1 score, classification reports, and confusion matrices rather than accuracy alone.

## Full-Dataset Results

Using the broader feature set, the strongest tree-based models produced extremely high classification performance:

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting | 99.74% | 100.00% | 98.96% | 99.48% |
| XGBoost | 99.74% | 100.00% | 98.96% | 99.48% |
| Random Forest | 99.74% | 100.00% | 98.96% | 99.48% |
| Logistic Regression | 92.95% | 78.00% | 100.00% | 87.64% |

The unusually strong results led to a closer investigation of whether some predictors were revealing information about the outcome rather than representing information that would actually be known when making a prediction.

## Feature Availability and Pre-Booking Analysis

The notebook then builds a second modeling dataset that removes variables identified as potentially dependent on the completed ride or cancellation outcome. This includes post-ride measures and payment-method indicators whose missing/unknown status was strongly tied to cancelled rides.

This experiment asks a more realistic modeling question:

> **Can a cancellation be predicted using information available before the ride is completed or cancelled?**

For Logistic Regression, performance fell substantially under the restricted feature set:

| Scenario | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Full-data Logistic Regression | 92.95% | 78.00% | 100.00% | 87.64% |
| Pre-booking Logistic Regression | 53.00% | 27.00% | 51.00% | 35.00% |

The notebook also finds that the tree-based ensemble models continue to perform strongly on the pre-booking feature set, while simpler models show a much larger decline. This reinforces the importance of comparing algorithms while also examining **when features become available and what information they encode**.

## Key Takeaways

- Missing values can carry meaningful information about the process that generated the data rather than simply representing errors to remove.
- Strong test-set performance should be interpreted in the context of feature availability.
- Restricting predictors to information available before an outcome can materially change model performance.
- Logistic Regression was much more sensitive to the pre-booking restriction, while tree-based ensemble methods captured stronger nonlinear relationships in the remaining predictors.
- Model evaluation benefits from looking beyond accuracy to precision, recall, F1 score, confusion matrices, and class balance.

## Tools and Technologies

**Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

**Methods:** Exploratory Data Analysis, Data Cleaning, Feature Encoding, Feature Scaling, Binary Classification, Model Comparison, Confusion Matrices, Precision/Recall/F1 Evaluation, Feature-Availability Analysis

## Repository Structure

```text
Uber-Cancellation-EDA-and-Modeling/
├── README.md
├── uber_ride_cancellation_analysis.ipynb
└── reports/
    └── project-report.pdf
```

## Project Files

- [View the Python analysis notebook](uber_ride_cancellation_analysis.ipynb)
- [View the project report](reports/project-report.pdf)

## Team Context

This was a two-person academic project. I was responsible for developing the complete Python notebook and technical analysis. My classmate collaborated on the written reporting and helped communicate the analysis and findings in the accompanying report.

## Limitations

This is an academic modeling project rather than a production deployment study. Several variables in the original dataset are only observed after a ride progresses or concludes, which is why the notebook explicitly compares the full dataset with a restricted pre-booking feature set. The results should be interpreted within those feature-availability constraints.