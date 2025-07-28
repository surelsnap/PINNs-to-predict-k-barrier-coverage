"""
Demo script for WSN Deployment Optimizer
This script demonstrates the key features and capabilities of the system.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.pinn_model import WSNPINNTrainer, create_pinn_model
from utils.data_processor import WSNDataProcessor

def main():
    """Main demo function."""
    
    print("🛰️ WSN Deployment Optimizer Demo")
    print("=" * 50)
    
    # 1. Load and process data
    print("\n📊 Loading and processing data...")
    processor = WSNDataProcessor()
    df = processor.load_data('data.csv')
    df_processed = processor.create_physics_features(df)
    
    print(f"Original dataset: {df.shape}")
    print(f"Processed dataset: {df_processed.shape}")
    print(f"Physics features created: {len(processor.physics_features)}")
    
    # 2. Show feature engineering
    print("\n🔧 Feature Engineering Results:")
    feature_sets = processor.get_feature_sets(df_processed)
    
    for name, features in feature_sets.items():
        if name != 'target':
            print(f"  {name.title()} features: {len(features)}")
            if len(features) <= 10:
                print(f"    {features}")
    
    # 3. Data analysis
    print("\n📈 Data Analysis:")
    target_stats = {
        'mean': df_processed['Number of Barriers'].mean(),
        'std': df_processed['Number of Barriers'].std(),
        'min': df_processed['Number of Barriers'].min(),
        'max': df_processed['Number of Barriers'].max()
    }
    
    print(f"  Target variable (Number of Barriers):")
    print(f"    Mean: {target_stats['mean']:.2f}")
    print(f"    Std: {target_stats['std']:.2f}")
    print(f"    Range: {target_stats['min']:.0f} - {target_stats['max']:.0f}")
    
    # 4. Show key relationships
    print("\n🔍 Key Feature Relationships:")
    
    # Coverage ratio vs barriers
    coverage_corr = df_processed['Coverage_Ratio'].corr(df_processed['Number of Barriers'])
    print(f"  Coverage Ratio vs Barriers: {coverage_corr:.3f}")
    
    # Sensor density vs barriers
    density_corr = df_processed['Sensor_Density'].corr(df_processed['Number of Barriers'])
    print(f"  Sensor Density vs Barriers: {density_corr:.3f}")
    
    # Transmission ratio vs barriers
    trans_corr = df_processed['Transmission_Ratio'].corr(df_processed['Number of Barriers'])
    print(f"  Transmission Ratio vs Barriers: {trans_corr:.3f}")
    
    # 5. Demo predictions
    print("\n🎯 Demo Predictions:")
    
    # Create a simple model for demo (without full training)
    features = feature_sets['combined']
    target = feature_sets['target']
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(
        df_processed, features, target, test_size=0.2, random_state=42
    )
    
    # Create and train a simple model
    print("  Training demo model...")
    model = create_pinn_model(input_dim=len(features))
    trainer = WSNPINNTrainer(model, learning_rate=0.01, lambda_physics=0.1)
    
    # Quick training for demo
    history = trainer.train(
        X_train, y_train, X_test, y_test, features,
        epochs=100, verbose=False
    )
    
    # Make predictions
    y_pred = trainer.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  Demo Model Performance:")
    print(f"    R² Score: {r2:.4f}")
    print(f"    MAE: {mae:.2f}")
    
    # 6. Show sample predictions
    print("\n📊 Sample Predictions:")
    sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        error = abs(actual - predicted)
        print(f"  Sample {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}, Error={error:.1f}")
    
    # 7. Demo deployment scenarios
    print("\n🚀 Demo Deployment Scenarios:")
    
    scenarios = [
        {
            'name': 'Small Office',
            'area': 10000,
            'sensing_range': 20,
            'transmission_range': 40,
            'num_sensors': 150
        },
        {
            'name': 'Large Warehouse',
            'area': 40000,
            'sensing_range': 30,
            'transmission_range': 60,
            'num_sensors': 300
        },
        {
            'name': 'Smart City District',
            'area': 50000,
            'sensing_range': 25,
            'transmission_range': 50,
            'num_sensors': 400
        }
    ]
    
    for scenario in scenarios:
        # Calculate physics features
        sensor_density = scenario['num_sensors'] / scenario['area']
        sensing_area = np.pi * scenario['sensing_range']**2
        total_sensing = scenario['num_sensors'] * sensing_area
        coverage_ratio = total_sensing / scenario['area']
        transmission_ratio = scenario['transmission_range'] / scenario['sensing_range']
        
        # Prepare input
        input_data = pd.DataFrame([{
            'Area': scenario['area'],
            'Sensing Range': scenario['sensing_range'],
            'Transmission Range': scenario['transmission_range'],
            'Number of Sensor nodes': scenario['num_sensors'],
            'Sensor_Density': sensor_density,
            'Sensing_Area_Per_Node': sensing_area,
            'Total_Sensing_Area': total_sensing,
            'Coverage_Ratio': coverage_ratio,
            'Transmission_Ratio': transmission_ratio,
            'Coverage_Efficiency': coverage_ratio / scenario['num_sensors'],
            'Connectivity_Proxy': transmission_ratio * sensor_density,
            'Area_Efficiency': scenario['num_sensors'] / np.sqrt(scenario['area'])
        }])
        
        # Predict
        prediction = trainer.predict(input_data)[0]
        barrier_percentage = (prediction / scenario['num_sensors']) * 100
        
        print(f"  {scenario['name']}:")
        print(f"    Predicted Barriers: {prediction:.1f}")
        print(f"    Barrier Percentage: {barrier_percentage:.1f}%")
        print(f"    Coverage Percentage: {100-barrier_percentage:.1f}%")
        
        # Recommendations
        if barrier_percentage > 20:
            print(f"    ⚠️  High barrier risk - consider increasing sensor density")
        elif barrier_percentage > 10:
            print(f"    ⚡ Moderate barrier risk - monitor deployment")
        else:
            print(f"    ✅ Good coverage - deployment looks optimal")
    
    # 8. System capabilities
    print("\n💡 System Capabilities:")
    capabilities = [
        "Physics-Informed Neural Networks for superior generalization",
        "Automated hyperparameter optimization with Optuna",
        "Interactive web application with Streamlit",
        "Real-time deployment optimization recommendations",
        "SHAP-based model explainability",
        "Comprehensive data analysis and visualization",
        "Production-ready model persistence and loading",
        "Scalable architecture for various deployment scenarios"
    ]
    
    for i, capability in enumerate(capabilities, 1):
        print(f"  {i}. {capability}")
    
    # 9. Real-world applications
    print("\n🌍 Real-World Applications:")
    applications = [
        "Smart City Infrastructure (traffic, environment, safety)",
        "Industrial IoT (manufacturing, warehouse, energy grid)",
        "Environmental Monitoring (forest fire, water quality, agriculture)",
        "Healthcare IoT (hospital monitoring, medical devices)",
        "Transportation (fleet management, logistics)",
        "Energy Management (smart grids, renewable energy)"
    ]
    
    for i, app in enumerate(applications, 1):
        print(f"  {i}. {app}")
    
    print("\n✅ Demo completed successfully!")
    print("\n🚀 To run the full system:")
    print("  1. python train_model.py    # Train the complete model")
    print("  2. streamlit run app.py     # Launch the web application")
    print("  3. Visit the web app for interactive deployment optimization")

if __name__ == "__main__":
    main() 