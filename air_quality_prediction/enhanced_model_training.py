import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime

# For preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path):
    """Load and perform initial data exploration"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def preprocess_data(df, target_col):
    """Preprocess data including handling missing values and feature engineering"""
    print("\n--- Data Preprocessing ---")
    
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Convert date columns if they exist
    date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
    for col in date_cols:
        if processed_df[col].dtype == 'object':
            try:
                processed_df[col] = pd.to_datetime(processed_df[col])
                # Extract useful date features
                processed_df[f'{col}_year'] = processed_df[col].dt.year
                processed_df[f'{col}_month'] = processed_df[col].dt.month
                processed_df[f'{col}_day'] = processed_df[col].dt.day
                processed_df[f'{col}_dayofweek'] = processed_df[col].dt.dayofweek
            except:
                print(f"Could not convert {col} to datetime. Keeping as is.")
    
    # Handle categorical data
    # First, identify categorical columns (object type or with less than 10 unique values)
    categorical_cols = [col for col in processed_df.columns 
                        if (processed_df[col].dtype == 'object' and col != target_col) 
                        or (processed_df[col].nunique() < 10 and col != target_col)]
    
    print(f"Categorical columns: {categorical_cols}")
    
    # Create dummies for categorical columns with few categories
    for col in categorical_cols:
        if processed_df[col].nunique() < 10:  # Only for columns with few categories
            dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
            processed_df = pd.concat([processed_df, dummies], axis=1)
            processed_df.drop(col, axis=1, inplace=True)
            print(f"Created dummies for {col}")
    
    # Handle missing values
    # For numerical columns: impute with median
    # For categorical columns: impute with mode
    numerical_cols = processed_df.select_dtypes(include=['number']).columns
    
    for col in numerical_cols:
        if processed_df[col].isnull().sum() > 0:
            median_val = processed_df[col].median()
            processed_df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    for col in processed_df.columns:
        if col not in numerical_cols and processed_df[col].isnull().sum() > 0:
            mode_val = processed_df[col].mode()[0]
            processed_df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")
    
    # Create interaction features for key pollutants
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    pollutant_cols = [col for col in pollutant_cols if col in processed_df.columns]
    
    if len(pollutant_cols) >= 2:
        # Create ratio features
        processed_df['PM_ratio'] = processed_df['PM2.5'] / processed_df['PM10'] if ('PM2.5' in processed_df.columns and 'PM10' in processed_df.columns) else np.nan
        
        # Create sum features
        if 'NO' in processed_df.columns and 'NO2' in processed_df.columns:
            processed_df['NOx_calculated'] = processed_df['NO'] + processed_df['NO2']
        
        # Create interaction terms
        if 'PM2.5' in processed_df.columns and 'O3' in processed_df.columns:
            processed_df['PM25_O3'] = processed_df['PM2.5'] * processed_df['O3']
    
    # Drop the original datetime columns after extraction
    for col in date_cols:
        if col in processed_df.columns:
            processed_df.drop(col, axis=1, inplace=True)
    
    # Drop any remaining non-numeric columns that couldn't be processed
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            print(f"Dropping non-numeric column: {col}")
            processed_df.drop(col, axis=1, inplace=True)
    
    return processed_df

def prepare_train_test(df, target_col, test_size=0.2):
    """Prepare training and testing datasets"""
    print("\n--- Preparing Training and Testing Sets ---")
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train multiple regression models and evaluate them"""
    print("\n--- Training Models ---")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Find the best model based on R2 score
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model Performance - RMSE: {best_metrics['rmse']:.4f}, MAE: {best_metrics['mae']:.4f}, R²: {best_metrics['r2']:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Best model ({best_model_name}) saved to models/best_model.pkl")
    
    return results, best_model_name, best_model

def tune_best_model(best_model_name, X_train_scaled, y_train, X_test_scaled, y_test):
    """Tune hyperparameters for the best model"""
    print(f"\n--- Tuning Hyperparameters for {best_model_name} ---")
    
    param_grids = {
        'Linear Regression': {},  # No hyperparameters to tune
        'Ridge Regression': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'ElasticNet': {
            'alpha': [0.01, 0.1, 1.0], 
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    }
    
    # Skip if the model doesn't have hyperparameters to tune
    if best_model_name not in param_grids or not param_grids[best_model_name]:
        print("No hyperparameters to tune for this model.")
        return None
    
    # Get the base model class
    base_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
    
    base_model = base_models[best_model_name]
    param_grid = param_grids[best_model_name]
    
    # Create and fit GridSearchCV
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (R²): {best_score:.4f}")
    
    # Evaluate on test set
    tuned_model = grid_search.best_estimator_
    y_pred = tuned_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Tuned Model Performance - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Save the tuned model
    joblib.dump(tuned_model, 'models/tuned_model.pkl')
    print("Tuned model saved to models/tuned_model.pkl")
    
    return tuned_model, best_params

def analyze_feature_importance(X_train, best_model, best_model_name):
    """Analyze feature importance for the model"""
    print("\n--- Feature Importance Analysis ---")
    
    # Models that support feature importance
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        # Get feature importances
        importances = best_model.feature_importances_
        
        # Create DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        print("Feature importance plot saved to models/feature_importance.png")
        
        return feature_importance_df
    
    elif best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        # Get coefficients
        coefficients = best_model.coef_
        
        # Create DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': coefficients
        })
        
        # Sort by absolute coefficient value
        feature_importance_df['Abs_Coefficient'] = np.abs(feature_importance_df['Coefficient'])
        feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)
        
        print("Top 10 features by coefficient magnitude:")
        print(feature_importance_df[['Feature', 'Coefficient']].head(10))
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df.head(15))
        plt.title(f'Feature Coefficients - {best_model_name}')
        plt.tight_layout()
        plt.savefig('models/feature_coefficients.png')
        print("Feature coefficients plot saved to models/feature_coefficients.png")
        
        return feature_importance_df
    
    else:
        print(f"Feature importance analysis not supported for {best_model_name}")
        return None

def main():
    """Main function to run the end-to-end model training pipeline"""
    print("=== Air Quality Prediction Model Training ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set file path
    file_path = 'updated_pollution_dataset.csv'
    
    # Target column to predict - can be changed as needed
    target_col = 'AQI'  # or 'PM2.5' or other pollutant
    
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    processed_df = preprocess_data(df, target_col)
    
    # Prepare training and testing sets
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test(processed_df, target_col)
    
    # Train models
    results, best_model_name, best_model = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Tune best model
    tuned_model, best_params = tune_best_model(best_model_name, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(X_train, best_model if tuned_model is None else tuned_model, best_model_name)
    
    print("\n=== Model Training Complete ===")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Models saved in the 'models' directory")

if __name__ == "__main__":
    main()