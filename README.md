# 🚗 French Road Accident Risk Explorer

A comprehensive web application for exploring and predicting French road-accident risk using historical accident data from 2015-2024. Built with Python and Streamlit for local-first data analysis.

## ✨ Features

### 📊 Overview Dashboard
- **Total Statistics**: View total accidents, fatalities, and serious injuries
- **Yearly Trends**: Analyze accident patterns from 2015-2024
- **Geographic Analysis**: Top departments by accident count
- **Temporal Patterns**: Day-of-week and hourly distributions
- **Environmental Factors**: Weather and lighting condition analysis

### 🔍 Interactive Filters
- **Dynamic Filtering**: Filter by year range, department, time, weather, and lighting
- **Real-time Updates**: Statistics and charts update instantly
- **Data Preview**: Browse filtered results in a table format

### 🎯 Risk Prediction
- **Machine Learning Model**: Random Forest classifier for serious accident prediction
- **Interactive Interface**: Select conditions and get risk estimates
- **Performance Metrics**: Model accuracy, AUC, and feature importance
- **Risk Visualization**: Gauge charts and probability displays

## 🛠️ Technology Stack

- **Frontend**: Streamlit (web framework)
- **Data Processing**: Pandas (data manipulation)
- **Machine Learning**: Scikit-learn (modeling)
- **Visualization**: Plotly (interactive charts)
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended (for large datasets)
- Local disk space for data files

## 🚀 Quick Start

### 1. Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd accidents_project

# Or download and extract the project folder
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Access the Application
Open your web browser and go to: **http://localhost:8501**

## 📁 Project Structure

```
accidents_project/
├── app.py                 # Main Streamlit application
├── data_loader.py         # Data loading and preprocessing utilities
├── modeling.py           # Machine learning model implementation
├── requirements.txt       # Python package dependencies
├── README.md             # This documentation file
├── accidents.ipynb       # Existing Jupyter notebook (if any)
└── data/                 # Accident data files (your existing structure)
    ├── 2015/
    │   └── 2015.csv
    ├── 2016/
    │   └── 2016.csv
    ├── ...
    ├── 2020/
    │   └── 2020.csv
    ├── 2021/
    │   ├── 2021.csv
    │   ├── carcteristiques-2021.csv
    │   ├── usagers-2021.csv
    │   ├── lieux-2021.csv
    │   └── vehicules-2021.csv
    ├── 2022/
    │   ├── 2022.csv
    │   ├── caracteristiques-2022.csv
    │   ├── usagers-2022.csv
    │   ├── lieux-2022.csv
    │   └── vehicules-2022.csv
    ├── 2023/
    │   ├── 2023.csv
    │   ├── caract-2023.csv
    │   ├── usagers-2023.csv
    │   ├── lieux-2023.csv
    │   └── vehicules-2023.csv
    └── 2024/
        ├── 2024.csv
        ├── caract-2024.csv
        ├── usagers-2024.csv
        ├── lieux-2024.csv
        └── vehicules-2024.csv
```

## 📊 Data Requirements

The application is designed to work with your existing French road accident data structure:

### Supported File Patterns
- **Yearly files**: `YYYY.csv`, `caracteristiques-YYYY.csv`, `caract-YYYY.csv`
- **Detailed tables**: `usagers-YYYY.csv`, `lieux-YYYY.csv`, `vehicules-YYYY.csv`
- **Flexible naming**: Handles variations like `carcteristiques-` vs `caract-`

### Automatic Detection
- **Separators**: Automatically detects `;` or `,` separators
- **Encodings**: Tries UTF-8, Latin-1, CP1252, ISO-8859-1
- **File Structure**: Adapts to different yearly organizations

### Key Data Fields Used
- **Temporal**: year, month, day, hour, time
- **Geographic**: department, commune
- **Environmental**: lighting (lum), weather (atm)
- **Road**: intersection (int), collision type (col)
- **Severity**: fatalities, serious injuries, gravity levels

## 🎯 Usage Guide

### Overview Page
1. **View Statistics**: Check total accidents, fatalities, and serious injury counts
2. **Analyze Trends**: Examine yearly patterns and trends
3. **Geographic Insights**: See which departments have the most accidents
4. **Temporal Patterns**: Understand accidents by day of week and hour
5. **Environmental Factors**: Analyze weather and lighting impacts

### Filters & Stats Page
1. **Apply Filters**: Use sidebar to filter data by various criteria
2. **Real-time Updates**: Watch statistics and charts update instantly
3. **Compare Scenarios**: Try different filter combinations
4. **Data Exploration**: Browse filtered results in the data table

### Prediction Page
1. **Train Model**: Click to train the risk prediction model (first-time use)
2. **Review Performance**: Check model accuracy and metrics
3. **Select Conditions**: Choose department, time, weather, and lighting
4. **Get Prediction**: View estimated serious accident risk
5. **Understand Factors**: See which conditions influence risk most

## ⚠️ Important Disclaimers

### Model Limitations
- **Historical Data Only**: Based on past accident patterns, not real-time conditions
- **Statistical Estimates**: Provides probabilities, not certainties
- **Educational Purpose**: For analysis and learning, not safety-critical decisions
- **Data Quality**: Accuracy depends on input data quality and completeness

### Usage Guidelines
- **❌ NOT for Real-time Safety**: Do not use for immediate safety decisions
- **✅ Educational Use**: Great for understanding accident patterns
- **✅ Research**: Suitable for academic and analytical purposes
- **✅ Planning**: Can inform general safety awareness and planning

### Risk Prediction
- Predicts **statistical likelihood** of serious accidents
- Based on **historical patterns** from 2015-2024
- **Not deterministic**: Same conditions can have different outcomes
- **Confidence intervals**: Consider uncertainty in predictions

## 🔧 Customization

### Adding New Data
1. Place CSV files in appropriate year folders
2. Follow existing naming conventions when possible
3. Application will automatically detect and load new files

### Model Adjustments
- Edit `modeling.py` to change algorithm parameters
- Modify feature selection in `prepare_data()` method
- Adjust training data timeframe as needed

### UI Customization
- Modify `app.py` for layout changes
- Add new pages or charts as desired
- Update styling in the CSS section

## 🐛 Troubleshooting

### Common Issues

#### "No data available" Error
- **Check**: Ensure CSV files are in correct folders
- **Verify**: File names match expected patterns
- **Check**: File permissions and accessibility

#### Model Training Fails
- **Data Quality**: Ensure sufficient data for training
- **Memory**: Check available RAM for large datasets
- **Features**: Verify required columns exist in data

#### Slow Performance
- **Data Size**: Consider filtering to recent years
- **Memory**: Close other applications to free RAM
- **Caching**: Restart app to clear cache if needed

#### Display Issues
- **Browser**: Try refreshing the page
- **Compatibility**: Use modern browser (Chrome, Firefox, Safari)
- **Network**: Ensure localhost is accessible

### Getting Help
1. **Check Console**: Look for error messages in terminal
2. **Verify Data**: Ensure data files are properly formatted
3. **Restart**: Try restarting the application
4. **Dependencies**: Confirm all packages are installed correctly

## 📈 Performance Tips

### For Large Datasets
- **Filter Early**: Use year filters to reduce data size
- **Memory Management**: Close unused browser tabs
- **Data Sampling**: Consider using subset for testing

### Optimization
- **Caching**: Application uses Streamlit caching automatically
- **Preprocessing**: Data is processed once and cached
- **Model Training**: Model is trained once per session

## 🤝 Contributing

This is designed as a personal data analysis tool. When modifying:
1. **Backup Data**: Keep original data files safe
2. **Test Changes**: Verify functionality with small data samples first
3. **Documentation**: Update documentation for any major changes
4. **Privacy**: Ensure compliance with data protection regulations

## 📄 License

This project is for educational and analytical purposes only. Use responsibly and in accordance with applicable data protection laws and regulations.

## 📞 Support

For technical issues:
1. Check this README for solutions
2. Verify all dependencies are installed
3. Ensure data files are properly structured
4. Check browser console for error messages

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Built with**: Streamlit, Python, and modern data science tools
