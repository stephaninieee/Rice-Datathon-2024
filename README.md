# Technical Report for Rice Datathon 2024: Predicting Peak Oil Production Rate (Chevron Challenge)
## Authors
- Oscar Wu, Judy Fang, Naman Gupta, Stephanie Chu

## Youtube Link
- https://youtu.be/IpAfl0Oqrq0

## Summary
- This report outlines the development and evaluation of a machine learning model to predict peak oil production rates using various geological and operational features. The project involves extensive data preprocessing, feature engineering, exploratory data analysis (EDA), model training, hyperparameter tuning, and model evaluation.

## Methodology
### Data Preprocessing
- **Data Loading**: The dataset is loaded from a CSV file.
- **Missing Value Handling**: The K-Nearest Neighbors (KNN) imputer is used to handle missing numerical value types in the dataset.

### Exploratory Data Analysis (EDA)
- **Data Distribution Analysis**: Histograms and boxplots are used to understand the distribution of various features and the target variable, `OilPeakRate`.
- **Correlation Analysis**: A correlation matrix is computed and visualized using a heatmap to identify potential relationships between features.

### Feature Selection
- **Feature Engineering**: We engineered four new features to further enhance the model's predictive power:
    - Well Trajectory Length (3D): 3D well length measure consisting of four existing features (`surface_x, surface_y, bh_x, bh_y, true_vertical_depth`)
        - `wall_trajectory_length` = \(\sqrt{(surface\_x - bh\_x)^2 + (surface\_y - bh\_y)^2 + (true\_vertical\_depth)^2}\)
    - Proppant & Fluid Efficiency: Resource usage efficiency for proppant and fluid.
        - `proppant_efficiency` = `total_proppant/true_vertical_depth`
        - `fluid_efficiency` = `total_fluid/true_vertical_depth`
    - Inclination: Offers insights into the well's drilling orientation, impacting its efficiency in accessing oil.
        - `wall_inclination` = \(\arccos(\frac{true\_vertical\_depth}{\text{Well Trajectory Length}})\)
- **Scaling**: After visualizing the distributions of the numerical features and target variable, we scaled each of them using either:
    - Min-Max scaling for features for minimal outliers
    - Robust scaling for features for outliers
- **Feature Importance Analysis**: Mutual Information Scores are calculated to identify the most predictive features.
- **Correlation-Based Feature Selection**: Features highly correlated with the target variable `OilPeakRate` are selected for model training.

### Model Development and Evaluation
- **Train-Test Split**: The dataset is split into training and test sets using a 80-20 split.
- **Baseline Model Training**: To measure baseline performance, we instantiated 7 machine learning models for comparison:
    - Linear Regression
    - CatBoostRegressor
    - GradientBoostingRegressor,
    - LGBMRegressor
    - RandomForestRegressor
    - DecisionTreeRegressor
    - Artificial Neural Network (ANN)
- **Model Evaluation**: We evaluated the baseline model performance for each of these 7 models using the Root Mean Squared Error (RMSE) metric.

### Hyperparameter Tuning
- **Tuning of LightGBM, CatBoost, and RandomForest Models**: RandomizedSearchCV and GridSearchCV are used to optimize model parameters, aiming to improve performance.

### Test Data Preprocessing
- **Preprocessing Function**: A function `process_csv` is defined to preprocess the test data in the same way as the training data.

### Predictions
- **Model Prediction**: The CatBoostRegressor model is used to make predictions on the test data.

## Results
- **Feature Engineering**: The creation of new features based on the given data improved the model's performance.
- **Model Comparison**: The CatBoostRegressor outperformed other models in terms of RMSE.
- **Hyperparameter Tuning**: The performance of models was enhanced post-tuning, with CatBoostRegressor showing significant improvement.
- **Test Data Prediction**: The CatBoostRegressor model was finalized for predicting peak oil rates on the test dataset and saved using the `joblib` library.

## Conclusion
- The project successfully demonstrates the application of machine learning techniques to predict peak oil production rates.
- The CatBoostRegressor, after hyperparameter tuning, emerged as the most effective model. - This model can be instrumental for asset development teams in predicting peak oil production rates, thereby aiding strategic decision-making in oil exploration and production.
