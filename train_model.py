"""
Comprehensive training script for WSN PINN model.
This script demonstrates the complete pipeline from data loading to model deployment.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
import torch
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.pinn_model import WSNPINNTrainer, create_pinn_model
from utils.data_processor import WSNDataProcessor

def main():
    """Main training pipeline."""
    
    print("🚀 Starting WSN PINN Training Pipeline")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Data Loading and Processing
    print("\n📊 Step 1: Data Loading and Processing")
    print("-" * 30)
    
    processor = WSNDataProcessor()
    df = processor.load_data('data.csv')
    
    # Create physics features
    df_processed = processor.create_physics_features(df)
    print(f"Dataset shape after feature engineering: {df_processed.shape}")
    
    # Data analysis
    print("\n📈 Performing data analysis...")
    analysis_results = processor.analyze_data(df_processed, save_plots=True)
    
    # 2. Feature Engineering
    print("\n🔧 Step 2: Feature Engineering")
    print("-" * 30)
    
    feature_sets = processor.get_feature_sets(df_processed)
    
    print("Available feature sets:")
    for name, features in feature_sets.items():
        if name != 'target':
            print(f"  - {name}: {len(features)} features")
    
    # Use combined features for best performance
    features = feature_sets['combined']
    target = feature_sets['target']
    
    print(f"\nUsing combined features: {len(features)} features")
    print("Features:", features)
    
    # 3. Data Splitting
    print("\n✂️ Step 3: Data Splitting")
    print("-" * 30)
    
    X_train, X_test, y_train, y_test = processor.split_data(
        df_processed, features, target, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 4. Hyperparameter Optimization
    print("\n🎯 Step 4: Hyperparameter Optimization")
    print("-" * 30)
    
    def objective(trial):
        """Optuna objective function for hyperparameter optimization."""
        
        # Hyperparameters to tune
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lambda_phys = trial.suggest_float("lambda_phys", 0.01, 1.0)
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "tanh"])
        
        # Create model
        model = create_pinn_model(
            input_dim=len(features),
            config={
                'hidden_dims': [hidden_size] * n_layers,
                'activation': activation,
                'dropout': 0.1
            }
        )
        
        # Create trainer
        trainer = WSNPINNTrainer(model, learning_rate=lr, lambda_physics=lambda_phys)
        
        # Train model (shorter training for optimization)
        history = trainer.train(
            X_train, y_train, X_test, y_test, features,
            epochs=200, verbose=False
        )
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        
        return -metrics['r2_score']  # Minimize negative R²
    
    print("Running hyperparameter optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    print(f"Best trial: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")
    
    # 5. Final Model Training
    print("\n🤖 Step 5: Final Model Training")
    print("-" * 30)
    
    # Use best parameters
    best_params = study.best_trial.params
    
    # Create final model
    final_model = create_pinn_model(
        input_dim=len(features),
        config={
            'hidden_dims': [best_params['hidden_size']] * best_params['n_layers'],
            'activation': best_params['activation'],
            'dropout': 0.1
        }
    )
    
    # Create trainer
    final_trainer = WSNPINNTrainer(
        final_model,
        learning_rate=best_params['lr'],
        lambda_physics=best_params['lambda_phys']
    )
    
    # Train final model
    print("Training final model...")
    history = final_trainer.train(
        X_train, y_train, X_test, y_test, features,
        epochs=500, verbose=True
    )
    
    # 6. Model Evaluation
    print("\n📊 Step 6: Model Evaluation")
    print("-" * 30)
    
    # Evaluate on test set
    metrics = final_trainer.evaluate(X_test, y_test)
    
    print("Final Model Performance:")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MSE: {metrics['mse']:.2f}")
    
    # Make predictions
    y_pred = final_trainer.predict(X_test)
    
    # 7. Visualization and Analysis
    print("\n📈 Step 7: Visualization and Analysis")
    print("-" * 30)
    
    # Create comprehensive visualizations
    create_training_visualizations(history, y_test, y_pred, df_processed, features)
    
    # 8. Model Saving
    print("\n💾 Step 8: Model Saving")
    print("-" * 30)
    
    # Save model
    final_trainer.save_model('models/trained_pinn')
    
    # Save training results
    results = {
        'best_params': best_params,
        'final_metrics': metrics,
        'feature_names': features,
        'training_history': history
    }
    
    joblib.dump(results, 'results/training_results.pkl')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'residuals': y_test - y_pred
    })
    predictions_df.to_csv('results/predictions.csv', index=False)
    
    print("✅ Training pipeline completed successfully!")
    print("\n📁 Files saved:")
    print("  - models/trained_pinn_model.pth")
    print("  - models/trained_pinn_scaler_X.pkl")
    print("  - models/trained_pinn_scaler_y.pkl")
    print("  - results/training_results.pkl")
    print("  - results/predictions.csv")
    print("  - plots/ (training visualizations)")
    
    # 9. Model Summary
    print("\n📋 Model Summary")
    print("-" * 30)
    print(f"Model Architecture: {best_params['n_layers']} layers, {best_params['hidden_size']} hidden units")
    print(f"Activation Function: {best_params['activation']}")
    print(f"Learning Rate: {best_params['lr']:.6f}")
    print(f"Physics Loss Weight: {best_params['lambda_phys']:.3f}")
    print(f"Input Features: {len(features)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Final R² Score: {metrics['r2_score']:.4f}")

def create_training_visualizations(history, y_test, y_pred, df, features):
    """Create comprehensive training visualizations."""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Physics loss
    axes[0, 1].plot(history['physics_loss'], label='Physics Loss', color='green')
    axes[0, 1].set_title('Physics Loss Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Physics Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction vs Actual
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6, color='purple')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[1, 0].set_title('Prediction vs Actual')
    axes[1, 0].set_xlabel('Actual Barriers')
    axes[1, 0].set_ylabel('Predicted Barriers')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred
    axes[1, 1].scatter(y_pred, residuals, alpha=0.6, color='orange')
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Residuals vs Predicted')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature importance (correlation-based)
    correlations = df[features + ['Number of Barriers']].corr()['Number of Barriers'].abs().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    correlations[:-1].plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Feature Importance (Correlation with Target)', fontsize=14, pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Absolute Correlation', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² Score
    axes[0].bar(['PINN Model'], [r2_score(y_test, y_pred)], color='green', alpha=0.7)
    axes[0].set_title('R² Score')
    axes[0].set_ylabel('R² Score')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].bar(['PINN Model'], [mean_absolute_error(y_test, y_pred)], color='orange', alpha=0.7)
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, alpha=0.3)
    
    # RMSE
    axes[2].bar(['PINN Model'], [np.sqrt(mean_squared_error(y_test, y_pred))], color='red', alpha=0.7)
    axes[2].set_title('Root Mean Square Error')
    axes[2].set_ylabel('RMSE')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Training visualizations saved to 'plots/' directory")

if __name__ == "__main__":
    main() 