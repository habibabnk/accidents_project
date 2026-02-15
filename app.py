import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_accident_data, debug_data_directory
from modeling import train_risk_model, predict_accident_risk, get_risk_model

# Page configuration
st.set_page_config(
    page_title="French Road Accident Risk Explorer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data (deprecated, use load_accident_data directly)"""
    return None

def create_overview_page(data):
    """Create overview page with statistics and charts"""
    st.markdown('<h1 class="main-header">📊 Accident Overview</h1>', unsafe_allow_html=True)
    
    if data is None:
        st.error("No data available")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_accidents = len(data)
        st.metric("Total Accidents", f"{total_accidents:,}")
    
    with col2:
        if 'fatalities' in data.columns:
            total_fatalities = data['fatalities'].fillna(0).sum()
            st.metric("Total Fatalities", f"{total_fatalities:,}")
        else:
            st.metric("Total Fatalities", "Data not available")
    
    with col3:
        if 'serious_injuries' in data.columns:
            total_serious = data['serious_injuries'].fillna(0).sum()
            st.metric("Serious Injuries", f"{total_serious:,}")
        else:
            st.metric("Serious Injuries", "Data not available")
    
    with col4:
        if 'is_serious' in data.columns:
            serious_rate = (data['is_serious'].mean() * 100)
            st.metric("Serious Accident Rate", f"{serious_rate:.1f}%")
        else:
            st.metric("Serious Accident Rate", "Data not available")
    
    # Trends by year
    st.subheader("📈 Accident Trends by Year")
    
    if 'year' in data.columns:
        yearly_stats = data.groupby('year').agg({
            'accident_id': 'count' if 'accident_id' in data.columns else lambda x: len(x),
            'is_serious': 'mean' if 'is_serious' in data.columns else lambda x: 0,
            'fatalities': 'sum' if 'fatalities' in data.columns else lambda x: 0
        }).reset_index()
        
        yearly_stats.columns = ['Year', 'Total Accidents', 'Serious Rate', 'Total Fatalities']
        yearly_stats['Serious Rate'] = yearly_stats['Serious Rate'] * 100
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Accidents', 'Serious Accident Rate (%)', 'Total Fatalities', 'Yearly Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total accidents
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total Accidents'], 
                      mode='lines+markers', name='Total Accidents'),
            row=1, col=1
        )
        
        # Serious rate
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Serious Rate'], 
                      mode='lines+markers', name='Serious Rate (%)'),
            row=1, col=2
        )
        
        # Fatalities
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total Fatalities'], 
                      mode='lines+markers', name='Total Fatalities'),
            row=2, col=1
        )
        
        # Combined comparison
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total Accidents'], 
                      mode='lines+markers', name='Accidents', yaxis='y4'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Department analysis
    if 'department' in data.columns:
        st.subheader("🗺️ Top Departments by Accident Count")
        
        dept_stats = data['department'].value_counts().head(15)
        
        fig = px.bar(
            x=dept_stats.values, 
            y=dept_stats.index,
            orientation='h',
            title='Top 15 Departments',
            labels={'x': 'Number of Accidents', 'y': 'Department'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal patterns
    col1, col2 = st.columns(2)
    
    with col1:
        if 'day_of_week' in data.columns:
            st.subheader("📅 Accidents by Day of Week")
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats = data['day_of_week'].value_counts().sort_index()
            day_stats.index = [day_names[i] for i in day_stats.index if i < len(day_names)]
            
            fig = px.bar(
                x=day_stats.index,
                y=day_stats.values,
                labels={'x': 'Day of Week', 'y': 'Number of Accidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'hour' in data.columns:
            st.subheader("🕐 Accidents by Hour of Day")
            
            hour_stats = data['hour'].value_counts().sort_index()
            
            fig = px.line(
                x=hour_stats.index,
                y=hour_stats.values,
                labels={'x': 'Hour of Day', 'y': 'Number of Accidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Environmental conditions
    col1, col2 = st.columns(2)
    
    with col1:
        if 'weather' in data.columns:
            st.subheader("🌤️ Accidents by Weather Conditions")
            
            weather_stats = data['weather'].value_counts().head(10)
            
            fig = px.pie(
                values=weather_stats.values,
                names=weather_stats.index,
                title='Weather Conditions'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'lighting' in data.columns:
            st.subheader("💡 Accidents by Lighting Conditions")
            
            lighting_stats = data['lighting'].value_counts().head(10)
            
            fig = px.pie(
                values=lighting_stats.values,
                names=lighting_stats.index,
                title='Lighting Conditions'
            )
            st.plotly_chart(fig, use_container_width=True)

def create_filters_page(data):
    """Create filters and statistics page"""
    st.markdown('<h1 class="main-header">🔍 Filters & Statistics</h1>', unsafe_allow_html=True)
    
    if data is None:
        st.error("No data available")
        return
    
    # Sidebar filters
    st.sidebar.subheader("📊 Data Filters")
    
    filtered_data = data.copy()
    
    # Year range filter
    if 'year' in data.columns:
        min_year = int(data['year'].min())
        max_year = int(data['year'].max())
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        filtered_data = filtered_data[
            (filtered_data['year'] >= year_range[0]) & 
            (filtered_data['year'] <= year_range[1])
        ]
    
    # Department filter
    if 'department' in data.columns:
        departments = sorted(data['department'].unique())
        selected_departments = st.sidebar.multiselect(
            "Departments",
            departments,
            default=departments[:10]  # Default to first 10
        )
        if selected_departments:
            filtered_data = filtered_data[filtered_data['department'].isin(selected_departments)]
    
    # Hour range filter
    if 'hour' in data.columns:
        min_hour = int(data['hour'].min())
        max_hour = int(data['hour'].max())
        hour_range = st.sidebar.slider(
            "Hour Range",
            min_value=min_hour,
            max_value=max_hour,
            value=(min_hour, max_hour)
        )
        filtered_data = filtered_data[
            (filtered_data['hour'] >= hour_range[0]) & 
            (filtered_data['hour'] <= hour_range[1])
        ]
    
    # Weather filter
    if 'weather' in data.columns:
        weather_options = sorted(data['weather'].unique())
        selected_weather = st.sidebar.multiselect(
            "Weather Conditions",
            weather_options,
            default=weather_options
        )
        if selected_weather:
            filtered_data = filtered_data[filtered_data['weather'].isin(selected_weather)]
    
    # Lighting filter
    if 'lighting' in data.columns:
        lighting_options = sorted(data['lighting'].unique())
        selected_lighting = st.sidebar.multiselect(
            "Lighting Conditions",
            lighting_options,
            default=lighting_options
        )
        if selected_lighting:
            filtered_data = filtered_data[filtered_data['lighting'].isin(selected_lighting)]
    
    # Display filtered statistics
    st.subheader("📈 Filtered Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Accidents", f"{len(filtered_data):,}")
    
    with col2:
        percentage = (len(filtered_data) / len(data)) * 100
        st.metric("Percentage of Total", f"{percentage:.1f}%")
    
    with col3:
        if 'is_serious' in filtered_data.columns:
            serious_rate = (filtered_data['is_serious'].mean() * 100)
            st.metric("Serious Rate", f"{serious_rate:.1f}%")
    
    with col4:
        if 'fatalities' in filtered_data.columns:
            fatalities = filtered_data['fatalities'].fillna(0).sum()
            st.metric("Fatalities", f"{fatalities:,}")
    
    # Filtered charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'department' in filtered_data.columns:
            st.subheader("📍 Department Distribution")
            
            dept_stats = filtered_data['department'].value_counts().head(10)
            
            fig = px.bar(
                x=dept_stats.values,
                y=dept_stats.index,
                orientation='h',
                labels={'x': 'Number of Accidents', 'y': 'Department'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'hour' in filtered_data.columns:
            st.subheader("🕐 Hour Distribution")
            
            hour_stats = filtered_data['hour'].value_counts().sort_index()
            
            fig = px.line(
                x=hour_stats.index,
                y=hour_stats.values,
                labels={'x': 'Hour', 'y': 'Number of Accidents'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table preview
    st.subheader("📋 Data Preview")
    st.dataframe(filtered_data.head(1000))

def create_prediction_page(data):
    """Create risk prediction page"""
    st.markdown('<h1 class="main-header">🎯 Risk Prediction</h1>', unsafe_allow_html=True)
    
    # Warning disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Disclaimer:</strong> This prediction tool provides statistical estimates based on historical data only. 
        It should NOT be used for real-time safety decisions or as a substitute for professional risk assessment. 
        The predictions are for educational and analytical purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    if data is None:
        st.error("No data available for modeling")
        return
    
    # Check if model is trained
    model = get_risk_model()
    
    if not model.is_trained:
        st.subheader("🤖 Training Risk Model")
        
        with st.spinner("Training model... This may take a few minutes."):
            success, result, trained_model = train_risk_model(data)
            
            if success:
                st.success("✅ Model trained successfully!")
                
                # Display model performance
                metrics = result
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Model Accuracy", f"{metrics['accuracy']:.3f}")
                    st.metric("AUC Score", f"{metrics['auc']:.3f}")
                
                with col2:
                    st.text("Classification Report:")
                    st.text(metrics['classification_report'])
                
                # Feature importance
                if trained_model.get_feature_importance() is not None:
                    st.subheader("📊 Feature Importance")
                    
                    importance_df = trained_model.get_feature_importance().head(10)
                    
                    fig = px.bar(
                        x=importance_df['importance'],
                        y=importance_df['feature'],
                        orientation='h',
                        labels={'x': 'Importance', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ Model training failed: {result}")
                return
    
    # Risk prediction form
    st.subheader("🎲 Predict Accident Risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Department selection
        if 'department' in data.columns:
            departments = sorted(data['department'].unique())
            selected_department = st.selectbox("Department", departments)
        
        # Month selection
        if 'month' in data.columns:
            months = list(range(1, 13))
            selected_month = st.selectbox("Month", months, format_func=lambda x: f"Month {x}")
        
        # Day of week selection
        if 'day_of_week' in data.columns:
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            selected_day = st.selectbox("Day of Week", range(len(days)), format_func=lambda x: days[x])
    
    with col2:
        # Hour selection
        if 'hour' in data.columns:
            hours = list(range(24))
            selected_hour = st.selectbox("Hour", hours, format_func=lambda x: f"{x:02d}:00")
        
        # Weather selection
        if 'weather' in data.columns:
            weather_options = sorted(data['weather'].unique())
            selected_weather = st.selectbox("Weather", weather_options)
        
        # Lighting selection
        if 'lighting' in data.columns:
            lighting_options = sorted(data['lighting'].unique())
            selected_lighting = st.selectbox("Lighting", lighting_options)
    
    # Predict button
    if st.button("🔮 Predict Risk", type="primary"):
        # Prepare input conditions
        conditions = {
            'year': 2023,  # Use recent year
            'month': selected_month,
            'day_of_week': selected_day,
            'hour': selected_hour,
            'department': selected_department,
            'weather': selected_weather,
            'lighting': selected_lighting
        }
        
        try:
            # Make prediction
            prediction = predict_accident_risk(conditions)
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Serious Accident Risk",
                    f"{prediction['risk_percentage']:.1f}%"
                )
            
            with col2:
                risk_level = "Low" if prediction['risk_percentage'] < 20 else "Medium" if prediction['risk_percentage'] < 40 else "High"
                st.metric("Risk Level", risk_level)
            
            with col3:
                st.metric("Risk Class", "Serious" if prediction['risk_class'] == 1 else "Not Serious")
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction['risk_percentage'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Serious Accident Probability (%)"},
                delta = {'reference': 20},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgray"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Conditions used
            st.subheader("📋 Conditions Used")
            conditions_df = pd.DataFrame(list(conditions.items()), columns=['Condition', 'Value'])
            st.dataframe(conditions_df)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def create_about_page():
    """Create about page"""
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🚗 French Road Accident Risk Explorer
    
    This web application explores and predicts French road-accident risk using historical accident data from 2015-2024.
    
    ### 📊 Data Source
    - **Dataset**: French road accident database
    - **Years**: 2015-2024
    - **Source**: User-provided CSV files
    - **Update Frequency**: Historical data only (no real-time updates)
    
    ### 🛠️ Technology Stack
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas
    - **Machine Learning**: Scikit-learn
    - **Visualization**: Plotly
    - **Language**: Python
    
    ### 📈 Features
    
    #### 1. Overview Dashboard
    - Total accident statistics
    - Yearly trends and patterns
    - Geographic distribution by department
    - Temporal patterns (day of week, hour)
    - Environmental factors (weather, lighting)
    
    #### 2. Interactive Filters
    - Dynamic data filtering
    - Real-time statistics updates
    - Customizable visualizations
    
    #### 3. Risk Prediction Model
    - Machine learning-based risk assessment
    - Feature importance analysis
    - Interactive prediction interface
    
    ### 🤖 Model Details
    
    **Algorithm**: Random Forest Classifier
    - Handles non-linear relationships
    - Provides feature importance
    - Robust to outliers
    
    **Target Variable**: Serious accidents (fatalities or serious injuries)
    
    **Features Used**:
    - Temporal: year, month, day of week, hour
    - Geographic: department
    - Environmental: weather, lighting conditions
    - Road: intersection type, collision type
    
    **Performance Metrics**:
    - Accuracy: Model performance on test data
    - AUC: Area under ROC curve
    - Feature importance: Variable impact ranking
    
    ### ⚠️ Limitations & Disclaimers
    
    **Data Limitations**:
    - Historical data only (no real-time information)
    - May contain reporting biases
    - Limited to available variables
    - Quality varies by year and region
    
    **Model Limitations**:
    - Statistical predictions, not deterministic
    - Based on historical patterns
    - Cannot account for unforeseen events
    - May not capture all risk factors
    
    **Usage Limitations**:
    - **NOT for real-time safety decisions**
    - Educational and analytical purposes only
    - Consult official sources for safety planning
    - Professional risk assessment recommended for critical applications
    
    ### 🚀 Getting Started
    
    1. **Install Dependencies**:
       ```bash
       pip install -r requirements.txt
       ```
    
    2. **Run Application**:
       ```bash
       streamlit run app.py
       ```
    
    3. **Access**: Open http://localhost:8501 in your browser
    
    ### 📁 Project Structure
    ```
    accidents_project/
    ├── app.py                 # Main Streamlit application
    ├── data_loader.py         # Data loading and preprocessing
    ├── modeling.py           # Machine learning models
    ├── requirements.txt       # Python dependencies
    ├── README.md             # Project documentation
    └── data/                 # Accident data files
        ├── 2015/
        ├── 2016/
        ├── ...
        └── 2024/
    ```
    
    ### 🤝 Contributing
    
    This is a local-first application designed for personal data analysis. Ensure data privacy and compliance with local regulations when using accident data.
    
    ### 📞 Support
    
    For questions or issues:
    1. Check the data files are properly formatted
    2. Verify all dependencies are installed
    3. Ensure sufficient memory for large datasets
    4. Check console output for error messages
    
    ---
    
    **Last Updated**: 2024
    **Version**: 1.0.0
    **License**: Educational Use Only
    """)

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("🚗 Accident Risk Explorer")
    
    # Data directory configuration
    st.sidebar.subheader("📁 Data Directory")
    default_data_dir = r"C:\Users\ThikPad\Documents\accidents_project"
    data_dir = st.sidebar.text_input(
        "Data Directory Path",
        value=default_data_dir,
        help="Path to the folder containing year subfolders with accident CSV files"
    )
    
    # Show resolved path
    resolved_path = Path(data_dir).resolve()
    st.sidebar.info(f"📍 Resolved path: {resolved_path}")
    
    # Debug data directory
    if st.sidebar.button("🔍 Scan Data Directory", help="Show all CSV files found"):
        with st.sidebar.spinner("Scanning files..."):
            total_files, file_preview = debug_data_directory(data_dir)
            st.sidebar.success(f"Found {total_files} CSV files")
            if file_preview:
                st.sidebar.write("**First 30 files found:**")
                for i, file_path in enumerate(file_preview[:10], 1):
                    st.sidebar.write(f"{i}. {file_path}")
                if len(file_preview) > 10:
                    st.sidebar.write(f"... and {len(file_preview) - 10} more files")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Overview", "🔍 Filters & Stats", "🎯 Prediction", "ℹ️ About"]
    )
    
    # Load data with error handling
    data = None
    loading_errors = []
    
    try:
        with st.spinner(f"Loading data from {data_dir}..."):
            result = load_accident_data(data_dir)
            if isinstance(result, tuple):
                data, loading_errors = result
            else:
                data = result
                loading_errors = []
    
        if data is not None and len(data) > 0:
            st.success(f"✅ Successfully loaded {len(data):,} accident records")
        else:
            st.error("❌ No data could be loaded")
            
            # Show loading errors
            if loading_errors:
                st.subheader("🚨 Loading Errors")
                for i, error in enumerate(loading_errors[:20], 1):
                    st.error(f"**Error {i}:** {error['file']}")
                    st.code(f"Message: {error['error']}")
                    if 'separator' in error:
                        st.write(f"Attempted separator: {error['separator']}")
                    if 'encoding' in error:
                        st.write(f"Attempted encoding: {error['encoding']}")
                    st.write("---")
                
                if len(loading_errors) > 20:
                    st.warning(f"... and {len(loading_errors) - 20} more errors")
    
    except Exception as e:
        st.error(f"❌ Critical error during data loading: {e}")
        st.write("**Troubleshooting steps:**")
        st.write("1. Check that the data directory path is correct")
        st.write("2. Ensure CSV files exist in the year subfolders")
        st.write("3. Click 'Scan Data Directory' to verify file detection")
        
        # Show debug info
        st.subheader("🔍 Debug Information")
        try:
            total_files, file_preview = debug_data_directory(data_dir)
            st.write(f"**CSV files found:** {total_files}")
            if file_preview:
                st.write("**Sample files:**")
                for file_path in file_preview[:5]:
                    st.write(f"- {file_path}")
        except Exception as debug_e:
            st.error(f"Debug scan failed: {debug_e}")
    
    # Display selected page
    if page == "📊 Overview":
        create_overview_page(data)
    elif page == "🔍 Filters & Stats":
        create_filters_page(data)
    elif page == "🎯 Prediction":
        create_prediction_page(data)
    elif page == "ℹ️ About":
        create_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("🚗 French Road Accident Risk Explorer")
    st.sidebar.markdown("Built with Streamlit")

if __name__ == "__main__":
    main()
