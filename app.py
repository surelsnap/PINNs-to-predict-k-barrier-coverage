"""
Streamlit Web Application for WSN Deployment Optimizer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.pinn_model import WSNPINNTrainer, create_pinn_model
from utils.data_processor import WSNDataProcessor

# Page configuration
st.set_page_config(
    page_title="WSN Deployment Optimizer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    processor = WSNDataProcessor()
    df = processor.load_data('data.csv')
    df_processed = processor.create_physics_features(df)
    return df_processed, processor

@st.cache_resource
def load_model():
    """Load the trained PINN model."""
    try:
        # Check if model files exist
        model_path = 'models/trained_pinn_model.pth'
        scaler_x_path = 'models/trained_pinn_scaler_X.pkl'
        scaler_y_path = 'models/trained_pinn_scaler_y.pkl'
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path)):
            return None
            
        # Create model and trainer
        model = create_pinn_model(input_dim=12)  # 4 raw + 8 physics features
        trainer = WSNPINNTrainer(model)
        
        # Try to load pre-trained model
        trainer.load_model('models/trained_pinn')
        return trainer
    except Exception as e:
        st.warning(f"No pre-trained model found or error loading model: {str(e)}. Please train the model first.")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🛰️ WSN Deployment Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Physics-Informed Neural Network for Wireless Sensor Network Optimization")
    
    # Load data and model
    df, processor = load_data()
    model = load_model()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Analysis", "🤖 Model Training", "🎯 Deployment Optimizer", "📈 Performance Metrics", "🔍 Model Explainability"]
    )
    
    if page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 Model Training":
        show_model_training(df, processor)
    elif page == "🎯 Deployment Optimizer":
        show_deployment_optimizer(df, processor, model)
    elif page == "📈 Performance Metrics":
        show_performance_metrics(df, processor, model)
    elif page == "🔍 Model Explainability":
        show_model_explainability(df, processor, model)

def show_data_analysis(df):
    """Display data analysis page."""
    st.header("📊 Dataset Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Target Range", f"{df['Number of Barriers'].min():.0f} - {df['Number of Barriers'].max():.0f}")
    with col4:
        st.metric("Mean Barriers", f"{df['Number of Barriers'].mean():.1f}")
    
    # Data preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select features to plot
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Select features to visualize:",
        numeric_cols,
        default=['Area', 'Sensing Range', 'Number of Sensor nodes', 'Number of Barriers']
    )
    
    if selected_features:
        fig = make_subplots(
            rows=len(selected_features), cols=1,
            subplot_titles=selected_features,
            vertical_spacing=0.1
        )
        
        for i, feature in enumerate(selected_features, 1):
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, showlegend=False),
                row=i, col=1
            )
        
        fig.update_layout(height=200 * len(selected_features), title_text="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.subheader("Key Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            df, x='Coverage_Ratio', y='Number of Barriers',
            title='Coverage Ratio vs Barriers',
            color='Area', size='Number of Sensor nodes'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df, x='Sensor_Density', y='Number of Barriers',
            title='Sensor Density vs Barriers',
            color='Transmission_Ratio', size='Area'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_training(df, processor):
    """Display model training page."""
    st.header("🤖 Model Training")
    
    # Training configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_set = st.selectbox(
            "Feature Set:",
            ["combined", "raw", "physics"],
            format_func=lambda x: x.title()
        )
        
        epochs = st.slider("Training Epochs:", 100, 1000, 500)
        
    with col2:
        learning_rate = st.selectbox(
            "Learning Rate:",
            [0.001, 0.01, 0.1],
            index=1
        )
        
        lambda_physics = st.slider("Physics Loss Weight:", 0.01, 1.0, 0.1)
    
    # Model architecture
    st.subheader("Model Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hidden_layers = st.selectbox("Hidden Layers:", [1, 2, 3, 4], index=1)
    with col2:
        hidden_size = st.selectbox("Hidden Size:", [32, 64, 128, 256], index=1)
    with col3:
        activation = st.selectbox("Activation:", ["relu", "leaky_relu", "tanh"])
    
    # Training button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Get feature sets
            feature_sets = processor.get_feature_sets(df)
            features = feature_sets[feature_set]
            target = feature_sets['target']
            
            # Split data
            X_train, X_test, y_train, y_test = processor.split_data(
                df, features, target
            )
            
            # Create and train model
            model = create_pinn_model(
                input_dim=len(features),
                config={
                    'hidden_dims': [hidden_size] * hidden_layers,
                    'activation': activation,
                    'dropout': 0.1
                }
            )
            
            trainer = WSNPINNTrainer(
                model, 
                learning_rate=learning_rate,
                lambda_physics=lambda_physics
            )
            
            # Train model
            history = trainer.train(
                X_train, y_train, X_test, y_test, features,
                epochs=epochs, verbose=False
            )
            
            # Evaluate model
            metrics = trainer.evaluate(X_test, y_test)
            
            # Save model
            trainer.save_model('models/trained_pinn')
            
            # Clear the model cache to force reload
            load_model.clear()
            
            # Display results
            st.success("Model training completed! Please refresh the page or navigate to another tab to use the trained model.")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics['r2_score']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.2f}")
            with col3:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
            with col4:
                st.metric("MSE", f"{metrics['mse']:.2f}")
            
            # Training history
            st.subheader("Training History")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['train_loss'], name='Training Loss'))
            fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
            fig.add_trace(go.Scatter(y=history['physics_loss'], name='Physics Loss'))
            
            fig.update_layout(
                title="Training Loss Over Time",
                xaxis_title="Epoch",
                yaxis_title="Loss"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_deployment_optimizer(df, processor, model):
    """Display deployment optimizer page."""
    st.header("🎯 Deployment Optimizer")
    
    if model is None:
        st.error("Please train a model first in the Model Training page.")
        
        # Add a refresh button
        if st.button("🔄 Refresh Model"):
            load_model.clear()
            st.rerun()
        return
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 How to use:</h4>
        <p>Adjust the deployment parameters below to see how they affect barrier formation. 
        The model will predict the number of barriers and provide optimization recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Deployment parameters
    st.subheader("Deployment Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.slider("Area (sq units):", 5000, 50000, 25000, step=1000)
        sensing_range = st.slider("Sensing Range:", 15, 40, 25)
        num_sensors = st.slider("Number of Sensors:", 50, 500, 200, step=10)
    
    with col2:
        transmission_range = st.slider("Transmission Range:", 30, 80, 50)
        
        # Auto-calculate physics features
        sensor_density = num_sensors / area
        sensing_area_per_node = np.pi * sensing_range**2
        total_sensing_area = num_sensors * sensing_area_per_node
        coverage_ratio = total_sensing_area / area
        transmission_ratio = transmission_range / sensing_range
        
        st.markdown("**Calculated Physics Features:**")
        st.write(f"• Sensor Density: {sensor_density:.6f}")
        st.write(f"• Coverage Ratio: {coverage_ratio:.4f}")
        st.write(f"• Transmission Ratio: {transmission_ratio:.2f}")
    
    # Make prediction
    if st.button("🔮 Predict Barriers", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame([{
            'Area': area,
            'Sensing Range': sensing_range,
            'Transmission Range': transmission_range,
            'Number of Sensor nodes': num_sensors,
            'Sensor_Density': sensor_density,
            'Sensing_Area_Per_Node': sensing_area_per_node,
            'Total_Sensing_Area': total_sensing_area,
            'Coverage_Ratio': coverage_ratio,
            'Transmission_Ratio': transmission_ratio,
            'Coverage_Efficiency': coverage_ratio / num_sensors,
            'Connectivity_Proxy': transmission_ratio * sensor_density,
            'Area_Efficiency': num_sensors / np.sqrt(area)
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Barriers", f"{prediction:.1f}")
        
        with col2:
            barrier_percentage = (prediction / num_sensors) * 100
            st.metric("Barrier %", f"{barrier_percentage:.1f}%")
        
        with col3:
            coverage_percentage = (1 - barrier_percentage/100) * 100
            st.metric("Coverage %", f"{coverage_percentage:.1f}%")
        
        # Optimization recommendations
        st.subheader("🎯 Optimization Recommendations")
        
        recommendations = []
        
        if prediction > 50:
            recommendations.append("⚠️ **High barrier count detected!** Consider increasing sensor density or sensing range.")
        
        if coverage_ratio < 0.5:
            recommendations.append("📡 **Low coverage ratio!** Increase sensing range or number of sensors.")
        
        if transmission_ratio < 1.5:
            recommendations.append("📶 **Low transmission ratio!** Increase transmission range relative to sensing range.")
        
        if sensor_density < 0.005:
            recommendations.append("🔍 **Low sensor density!** Add more sensors or reduce deployment area.")
        
        if not recommendations:
            recommendations.append("✅ **Good deployment configuration!** Current parameters should provide adequate coverage.")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Parameter sensitivity analysis
        st.subheader("📊 Parameter Sensitivity Analysis")
        
        # Test variations
        variations = []
        base_params = {
            'Area': area, 'Sensing Range': sensing_range,
            'Transmission Range': transmission_range, 'Number of Sensor nodes': num_sensors
        }
        
        # Vary each parameter
        for param, value in base_params.items():
            for factor in [0.8, 1.0, 1.2]:
                test_params = base_params.copy()
                test_params[param] = value * factor
                
                # Calculate physics features for test params
                test_sensor_density = test_params['Number of Sensor nodes'] / test_params['Area']
                test_sensing_area = np.pi * test_params['Sensing Range']**2
                test_total_sensing = test_params['Number of Sensor nodes'] * test_sensing_area
                test_coverage_ratio = test_total_sensing / test_params['Area']
                test_transmission_ratio = test_params['Transmission Range'] / test_params['Sensing Range']
                
                test_data = pd.DataFrame([{
                    'Area': test_params['Area'],
                    'Sensing Range': test_params['Sensing Range'],
                    'Transmission Range': test_params['Transmission Range'],
                    'Number of Sensor nodes': test_params['Number of Sensor nodes'],
                    'Sensor_Density': test_sensor_density,
                    'Sensing_Area_Per_Node': test_sensing_area,
                    'Total_Sensing_Area': test_total_sensing,
                    'Coverage_Ratio': test_coverage_ratio,
                    'Transmission_Ratio': test_transmission_ratio,
                    'Coverage_Efficiency': test_coverage_ratio / test_params['Number of Sensor nodes'],
                    'Connectivity_Proxy': test_transmission_ratio * test_sensor_density,
                    'Area_Efficiency': test_params['Number of Sensor nodes'] / np.sqrt(test_params['Area'])
                }])
                
                test_prediction = model.predict(test_data)[0]
                variations.append({
                    'Parameter': param,
                    'Factor': factor,
                    'Value': test_params[param],
                    'Prediction': test_prediction
                })
        
        variations_df = pd.DataFrame(variations)
        
        # Plot sensitivity
        fig = px.line(
            variations_df, x='Factor', y='Prediction', color='Parameter',
            title='Parameter Sensitivity Analysis',
            labels={'Factor': 'Parameter Multiplier', 'Prediction': 'Predicted Barriers'}
        )
        fig.add_hline(y=prediction, line_dash="dash", line_color="red", 
                     annotation_text="Current Prediction")
        st.plotly_chart(fig, use_container_width=True)

def show_performance_metrics(df, processor, model):
    """Display performance metrics page."""
    st.header("📈 Performance Metrics")
    
    if model is None:
        st.error("Please train a model first in the Model Training page.")
        return
    
    # Get feature sets and split data
    feature_sets = processor.get_feature_sets(df)
    features = feature_sets['combined']
    target = feature_sets['target']
    
    X_train, X_test, y_train, y_test = processor.split_data(df, features, target)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mse': mean_squared_error(y_test, y_pred)
    }
    
    # Display metrics
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{metrics['r2_score']:.4f}")
    with col2:
        st.metric("MAE", f"{metrics['mae']:.2f}")
    with col3:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    with col4:
        st.metric("MSE", f"{metrics['mse']:.2f}")
    
    # Prediction vs Actual plot
    st.subheader("Prediction vs Actual")
    
    fig = px.scatter(
        x=y_test, y=y_pred,
        title="Model Predictions vs Actual Values",
        labels={'x': 'Actual Barriers', 'y': 'Predicted Barriers'}
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals plot
    st.subheader("Residuals Analysis")
    
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            x=y_pred, y=residuals,
            title="Residuals vs Predicted",
            labels={'x': 'Predicted Values', 'y': 'Residuals'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            x=residuals,
            title="Residuals Distribution",
            labels={'x': 'Residuals'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_explainability(df, processor, model):
    """Display model explainability page."""
    st.header("🔍 Model Explainability")
    
    if model is None:
        st.error("Please train a model first in the Model Training page.")
        return
    
    st.markdown("""
    <div class="info-box">
        <h4>🔍 SHAP Analysis:</h4>
        <p>SHAP (SHapley Additive exPlanations) values show how each feature contributes to the model's predictions. 
        This helps understand which factors are most important for barrier formation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance analysis
    st.subheader("Feature Importance Analysis")
    
    # Get feature sets
    feature_sets = processor.get_feature_sets(df)
    features = feature_sets['combined']
    
    # Sample data for SHAP analysis
    sample_data = df[features].sample(min(50, len(df)), random_state=42)
    
    # Calculate approximate feature importance using correlation
    correlations = df[features + ['Number of Barriers']].corr()['Number of Barriers'].abs().sort_values(ascending=False)
    
    # Display feature importance
    fig = px.bar(
        x=correlations.index[:-1],  # Exclude target variable
        y=correlations.values[:-1],
        title="Feature Importance (Correlation with Target)",
        labels={'x': 'Features', 'y': 'Absolute Correlation'}
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.subheader("Key Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Coverage ratio impact
        fig = px.scatter(
            df, x='Coverage_Ratio', y='Number of Barriers',
            title='Coverage Ratio Impact',
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sensor density impact
        fig = px.scatter(
            df, x='Sensor_Density', y='Number of Barriers',
            title='Sensor Density Impact',
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive feature analysis
    st.subheader("Interactive Feature Analysis")
    
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        features
    )
    
    if selected_feature:
        fig = px.scatter(
            df, x=selected_feature, y='Number of Barriers',
            title=f'{selected_feature} vs Number of Barriers',
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean", f"{df[selected_feature].mean():.4f}")
        with col2:
            st.metric("Std", f"{df[selected_feature].std():.4f}")
        with col3:
            st.metric("Correlation", f"{df[selected_feature].corr(df['Number of Barriers']):.4f}")

if __name__ == "__main__":
    main() 