## Note for each function and step

## Code needed to run:
- Activate environment:
    ```
    conda activate venv/
    ```
- Run file example:
    ```
    python -m src.components.data_ingestion
    ```






## Code Explaination

```
@dataclass
```
- The @dataclass decorator in Python (introduced in Python 3.7) streamlines the creation of classes that are primarily used to store data. Normally, when you define a class to hold data, you must manually write methods like __init__, __repr__, and __eq__ to properly initialize, display, and compare instances of the class. With @dataclass, these methods are automatically generated for you based on the class variables you define.



line 53, data_transformation.py
```
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
                
            )

```

Why do i have to specify "scaler", StandardScaler(with_mean=False) for categorical variable?
When using StandardScaler on categorical features that have been one-hot encoded, setting with_mean=False is often necessary to avoid unintended consequences:

One-Hot Encoded (OHE) Features are Sparse and Binary:
After one-hot encoding, categorical features become a series of binary columns (0s and 1s). These columns are not naturally centered around zero—they represent the presence (1) or absence (0) of a category. If you try to subtract the mean (which StandardScaler does by default), you’ll shift these binary values into potentially negative or fractional values, which may not be semantically meaningful for these features.

Preserving Data Representation:
For binary indicators:

Mean-centering would turn a feature that’s originally 0 or 1 into something like -0.2 or +0.8, depending on the distribution. This distorts the straightforward interpretation of these features and can make them less intuitive for models or for later analysis.
Avoiding Dense Matrices from Sparse Data:
One-hot encoded features are often sparse—most entries are zero. Subtracting the mean from each column effectively removes the sparse nature, as most entries will no longer be zero, potentially leading to less efficient computations and memory usage.

By setting with_mean=False, StandardScaler will only scale by the standard deviation without mean-centering. This approach keeps the structure of your OHE data simpler and avoids introducing negative or fractional values where they don’t make sense. It’s a pragmatic choice primarily made for convenience and interpretability in handling categorical features after one-hot encoding.



- line 103, data_transformation.py

```
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
```
'''
            In machine learning, the idea is that you learn all the data-driven parameters (such as means, standard deviations for scaling, or learned encodings for categorical variables) strictly from the training data, and then apply exactly the same transformations to the test data without re-learning those parameters. This is a critical step to maintain the integrity of your model evaluation and prevent data leakage.

            **Detailed Reasoning:**

                1. **What `fit_transform` Does on Training Data:**  
                When you call `fit_transform` on the training dataset, you are performing two operations:
                - **Fit:** The pipeline (or transformer) calculates any necessary statistics or encodings from the training set. For example:
                    - A `SimpleImputer` might learn the median value of a feature to fill in missing values.
                    - A `StandardScaler` will learn the mean and standard deviation of each numerical feature.
                    - A `OneHotEncoder` will learn the unique categories present in each categorical feature.
                    
                - **Transform:** After calculating those statistics and encodings, the transformer applies the transformation to the training data itself (e.g., scaling the features, encoding categories).

                By doing `fit_transform` on the training set, you end up with a transformed version of the training data and a fitted preprocessor that holds all the learned parameters (like means, medians, categories).

                2. **What `transform` Does on Test Data:**  
                When you call `transform` on the test set, the transformer does **not** re-learn parameters. Instead, it:
                - Uses the **already learned** statistics and encodings from the training data.
                - Applies these learned transformations to the new, unseen test data without modifying its internal parameters.

                This ensures that the test data is processed in the same way as the training data, using only the information gathered from training. If you were to `fit_transform` the test set, you’d be recalculating means, medians, categories, etc., from the test set as well—this would be a form of data leakage and would not give you a realistic evaluation of how your model performs on truly unseen data.

                **In short:**  
                - **`fit_transform` on training data:** Learn and apply transformations.  
                - **`transform` on test data:** Apply the already learned transformations without modifying them.

                This pattern ensures that your model evaluation on the test data reflects a true "out-of-sample" scenario, which is the essence of proper model validation.

            '''



# Model Selection

- ``` model_trainer.py```

    ```
    models = {
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Linear Regression": LinearRegression(),
    "XGBRegressor": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),                
    
}
    ```

    Summary:
    
    ### Random Forest
    - Characteristics: Handles non-linear relationships, robust to overfitting, interpretable via feature importance
    - Best Use Cases: Complex data with limited preprocessing

    ### Decision Tree
    - Characteristics: Simple and interpretable but prone to overfitting
    - Best Use Cases: Quick insights, small datasets
    
    
    ### Gradient Boosting
    - Characteristics: Sequential learning, good for complex patterns, prone to overfitting without tuning
    - Best Use Cases: Structured data with non-linear patterns
    
    ### Linear Regression
    - Characteristics: Assumes linearity, fast, interpretable
    - Best Use Cases: 	Linear relationships, baseline model
    
    
    ### XGBoost
    - Characteristics: Optimized boosting, high performance, handles missing values
    - Best Use Cases: Large datasets, competitive modeling
    
    ### CatBoost
    - Characteristics: Optimized for categorical data, less overfitting
    - Best Use Cases: Data with many categorical features

    

    ### AdaBoost	
    - Characteristics: Emphasizes difficult-to-predict samples, combines weak learners	
    - Best Use Cases: Small, clean datasets

