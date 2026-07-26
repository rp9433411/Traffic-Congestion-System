"""
Visualization Module for Traffic Congestion Prediction.

Generates professional visualizations:
- Congestion heatmaps (time vs location)
- Feature importance plots
- Model performance comparison
- Time series analysis
- Weather impact analysis
- Confusion matrices
- Prediction vs actual scatter plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, MODELS_DIR, CONGESTION_LEVELS, DATA_DIR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrafficVisualizer:
    """
    Creates professional visualizations for traffic congestion analysis.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or str(OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_congestion_heatmap(self, data: pd.DataFrame, 
                                save: bool = True) -> go.Figure:
        """
        Create interactive heatmap of congestion by hour and location.
        
        Args:
            data: DataFrame with congestion data
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        # Aggregate data
        if 'hour' in data.columns and 'location' in data.columns:
            pivot = data.pivot_table(
                values='congestion_level',
                index='location',
                columns='hour',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot,
                title='Traffic Congestion Heatmap: Location vs Hour of Day',
                labels=dict(x='Hour of Day', y='Location', color='Congestion Level'),
                color_continuous_scale='RdYlGn_r',
                aspect='auto',
                template='plotly_dark'
            )
            
            fig.update_layout(
                title={'font': {'size': 20, 'family': 'Arial'}},
                xaxis={'dtick': 1},
                height=600,
                width=1000
            )
            
            if save:
                fig.write_html(os.path.join(self.output_dir, 'congestion_heatmap.html'))
                fig.write_image(os.path.join(self.output_dir, 'congestion_heatmap.png'), 
                               scale=2)
                
            return fig
    
    def plot_feature_importance(self, model, feature_names: List[str],
                               model_name: str = 'Model', save: bool = True) -> go.Figure:
        """
        Plot feature importance from tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature_importances_")
            return None
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Top 15 features
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        fig = go.Figure(go.Bar(
            x=top_importances[::-1],
            y=top_features[::-1],
            orientation='h',
            marker=dict(
                color=top_importances[::-1],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Importance')
            ),
            text=[f'{imp:.1%}' for imp in top_importances[::-1]],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance - {model_name}',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_dark',
            height=600,
            width=900,
            margin=dict(l=200, r=50, t=50, b=50)
        )
        
        if save:
            fig.write_html(os.path.join(self.output_dir, f'feature_importance_{model_name}.html'))
            fig.write_image(os.path.join(self.output_dir, f'feature_importance_{model_name}.png'),
                           scale=2)
            
        return fig
    
    def plot_model_comparison(self, metrics: Dict, save: bool = True) -> go.Figure:
        """
        Create bar chart comparing model performance.
        
        Args:
            metrics: Dictionary of model metrics
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        models = list(metrics.keys())
        
        # Extract metrics
        mae_values = [metrics[m].get('MAE', 0) for m in models]
        rmse_values = [metrics[m].get('RMSE', 0) for m in models]
        r2_values = [metrics[m].get('R2', 0) for m in models]
        accuracy_values = [metrics[m].get('Accuracy', 0) for m in models]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)',
                          'R² Score', 'Accuracy'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # MAE subplot
        fig.add_trace(go.Bar(
            x=models, y=mae_values,
            marker_color=colors[:len(models)],
            text=[f'{v:.4f}' for v in mae_values],
            textposition='outside',
            showlegend=False
        ), row=1, col=1)
        
        # RMSE subplot
        fig.add_trace(go.Bar(
            x=models, y=rmse_values,
            marker_color=colors[:len(models)],
            text=[f'{v:.4f}' for v in rmse_values],
            textposition='outside',
            showlegend=False
        ), row=1, col=2)
        
        # R² subplot
        fig.add_trace(go.Bar(
            x=models, y=r2_values,
            marker_color=colors[:len(models)],
            text=[f'{v:.4f}' for v in r2_values],
            textposition='outside',
            showlegend=False
        ), row=2, col=1)
        
        # Accuracy subplot
        fig.add_trace(go.Bar(
            x=models, y=accuracy_values,
            marker_color=colors[:len(models)],
            text=[f'{v:.1%}' for v in accuracy_values],
            textposition='outside',
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(
            title='Model Performance Comparison',
            template='plotly_dark',
            height=800,
            width=1200,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'model_comparison.html'))
            fig.write_image(os.path.join(self.output_dir, 'model_comparison.png'),
                           scale=2)
            
        return fig
    
    def plot_congestion_distribution(self, data: pd.DataFrame, 
                                     save: bool = True) -> go.Figure:
        """
        Plot distribution of congestion levels.
        
        Args:
            data: DataFrame with congestion data
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if 'congestion_level' not in data.columns:
            return None
            
        # Count distribution
        dist = data['congestion_level'].value_counts().sort_index()
        
        labels = [CONGESTION_LEVELS.get(i, f'Level {i}') for i in dist.index]
        colors = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Congestion Level Distribution', 'Congestion Pie Chart'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        fig.add_trace(go.Bar(
            x=labels,
            y=dist.values,
            marker_color=colors[:len(dist)],
            text=dist.values,
            textposition='outside',
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=dist.values,
            marker_colors=colors[:len(dist)],
            hole=0.4,
            textinfo='label+percent'
        ), row=1, col=2)
        
        fig.update_layout(
            title='Congestion Level Distribution',
            template='plotly_dark',
            height=500,
            width=1200
        )
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'congestion_distribution.html'))
            fig.write_image(os.path.join(self.output_dir, 'congestion_distribution.png'),
                           scale=2)
            
        return fig
    
    def plot_time_series_analysis(self, data: pd.DataFrame, 
                                  save: bool = True) -> go.Figure:
        """
        Plot time series analysis of traffic metrics.
        
        Args:
            data: DataFrame with time series data
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Traffic Volume Over Time', 'Average Speed Over Time',
                          'Congestion Level Over Time'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Sort by timestamp if available
        plot_data = data.copy()
        if 'timestamp' in plot_data.columns:
            plot_data = plot_data.sort_values('timestamp')
            x = plot_data['timestamp']
        else:
            x = plot_data.index
            
        # Traffic volume
        if 'traffic_volume' in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=x[:500], y=plot_data['traffic_volume'][:500],
                mode='lines',
                name='Traffic Volume',
                line=dict(color='#3498DB', width=2)
            ), row=1, col=1)
            
        # Average speed
        if 'avg_speed' in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=x[:500], y=plot_data['avg_speed'][:500],
                mode='lines',
                name='Avg Speed',
                line=dict(color='#2ECC71', width=2)
            ), row=2, col=1)
            
        # Congestion level
        if 'congestion_level' in plot_data.columns:
            fig.add_trace(go.Scatter(
                x=x[:500], y=plot_data['congestion_level'][:500],
                mode='lines+markers',
                name='Congestion Level',
                line=dict(color='#E74C3C', width=2),
                marker=dict(size=4)
            ), row=3, col=1)
            
        fig.update_layout(
            title='Traffic Metrics Time Series Analysis',
            template='plotly_dark',
            height=900,
            width=1200,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text='Vehicles', row=1, col=1)
        fig.update_yaxes(title_text='km/h', row=2, col=1)
        fig.update_yaxes(title_text='Level (0-3)', row=3, col=1)
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'time_series_analysis.html'))
            fig.write_image(os.path.join(self.output_dir, 'time_series_analysis.png'),
                           scale=2)
            
        return fig
    
    def plot_weather_impact(self, data: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Plot impact of weather on congestion.
        
        Args:
            data: DataFrame with weather and congestion data
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Congestion vs Temperature', 'Congestion vs Precipitation',
                          'Congestion vs Humidity', 'Congestion vs Wind Speed'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Temperature vs Congestion
        if all(col in data.columns for col in ['temperature', 'congestion_level']):
            fig.add_trace(go.Scatter(
                x=data['temperature'], y=data['congestion_level'],
                mode='markers',
                name='Temperature',
                marker=dict(color='#FF6B6B', size=5, opacity=0.5),
                showlegend=False
            ), row=1, col=1)
            
        # Precipitation vs Congestion
        if all(col in data.columns for col in ['precipitation', 'congestion_level']):
            fig.add_trace(go.Scatter(
                x=data['precipitation'], y=data['congestion_level'],
                mode='markers',
                name='Precipitation',
                marker=dict(color='#4ECDC4', size=5, opacity=0.5),
                showlegend=False
            ), row=1, col=2)
            
        # Humidity vs Congestion
        if all(col in data.columns for col in ['humidity', 'congestion_level']):
            fig.add_trace(go.Scatter(
                x=data['humidity'], y=data['congestion_level'],
                mode='markers',
                name='Humidity',
                marker=dict(color='#45B7D1', size=5, opacity=0.5),
                showlegend=False
            ), row=2, col=1)
            
        # Wind Speed vs Congestion
        if all(col in data.columns for col in ['wind_speed', 'congestion_level']):
            fig.add_trace(go.Scatter(
                x=data['wind_speed'], y=data['congestion_level'],
                mode='markers',
                name='Wind Speed',
                marker=dict(color='#96CEB4', size=5, opacity=0.5),
                showlegend=False
            ), row=2, col=2)
            
        fig.update_layout(
            title='Weather Impact on Traffic Congestion',
            template='plotly_dark',
            height=800,
            width=1200
        )
        
        fig.update_xaxes(title_text='Temperature (°C)', row=1, col=1)
        fig.update_xaxes(title_text='Precipitation (mm)', row=1, col=2)
        fig.update_xaxes(title_text='Humidity (%)', row=2, col=1)
        fig.update_xaxes(title_text='Wind Speed (km/h)', row=2, col=2)
        fig.update_yaxes(title_text='Congestion Level', row=1, col=1)
        fig.update_yaxes(title_text='Congestion Level', row=1, col=2)
        fig.update_yaxes(title_text='Congestion Level', row=2, col=1)
        fig.update_yaxes(title_text='Congestion Level', row=2, col=2)
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'weather_impact.html'))
            fig.write_image(os.path.join(self.output_dir, 'weather_impact.png'),
                           scale=2)
            
        return fig
    
    def plot_rush_hour_analysis(self, data: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Plot rush hour analysis.
        
        Args:
            data: DataFrame with time and congestion data
            save: Whether to save the plot
            
        Returns:
            Plotly figure
        """
        if 'hour' not in data.columns or 'congestion_level' not in data.columns:
            return None
            
        # Average congestion by hour
        hourly_avg = data.groupby('hour')['congestion_level'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'],
            mode='lines+markers',
            name='Avg Congestion',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=10, color='#E74C3C')
        ))
        
        # Add standard deviation band
        fig.add_trace(go.Scatter(
            x=pd.concat([hourly_avg['hour'], hourly_avg['hour'][::-1]]),
            y=pd.concat([hourly_avg['mean'] + hourly_avg['std'],
                        hourly_avg['mean'] - hourly_avg['std'][::-1]]),
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Std Deviation',
            showlegend=True
        ))
        
        # Add rush hour markers
        fig.add_vrect(x0=7, x1=9, fillcolor='rgba(255, 0, 0, 0.1)', 
                      annotation_text='Morning Rush', annotation_position='top left')
        fig.add_vrect(x0=17, x1=19, fillcolor='rgba(255, 0, 0, 0.1)',
                      annotation_text='Evening Rush', annotation_position='top left')
        
        fig.update_layout(
            title='Rush Hour Analysis: Average Congestion by Hour of Day',
            xaxis=dict(title='Hour of Day', dtick=1),
            yaxis=dict(title='Average Congestion Level (0-3)'),
            template='plotly_dark',
            height=600,
            width=1000,
            hovermode='x'
        )
        
        if save:
            fig.write_html(os.path.join(self.output_dir, 'rush_hour_analysis.html'))
            fig.write_image(os.path.join(self.output_dir, 'rush_hour_analysis.png'),
                           scale=2)
            
        return fig
    
    def generate_all_visualizations(self, data: pd.DataFrame, 
                                   models: Dict = None,
                                   feature_names: List[str] = None,
                                   metrics: Dict = None):
        """
        Generate all visualizations.
        
        Args:
            data: DataFrame with traffic data
            models: Dictionary of trained models
            feature_names: List of feature names
            metrics: Dictionary of model metrics
        """
        print("Generating all visualizations...")
        
        # Basic analysis plots
        self.plot_congestion_distribution(data)
        self.plot_congestion_heatmap(data)
        self.plot_time_series_analysis(data)
        self.plot_weather_impact(data)
        self.plot_rush_hour_analysis(data)
        
        # Model-specific plots
        if models and feature_names:
            for name, model in models.items():
                self.plot_feature_importance(model, feature_names, name)
        
        if metrics:
            self.plot_model_comparison(metrics)
            
        print(f"All visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    # Test visualizations
    from src.data_generator import TrafficDataGenerator
    
    generator = TrafficDataGenerator()
    data = generator.generate_dataset(2000)
    
    visualizer = TrafficVisualizer()
    visualizer.generate_all_visualizations(data)
    
    print("Visualization test complete!")