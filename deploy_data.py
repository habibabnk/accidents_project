"""
Data loader for Streamlit Community Cloud deployment.
This script handles data loading for deployment when local CSV files are not available.
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime
import requests
import io

# URLs for sample data (you'll need to host these somewhere)
SAMPLE_DATA_URLS = {
    "2021": "https://raw.githubusercontent.com/habibabnk/accidents_project/main/data/2021/2021.csv",
    "2022": "https://raw.githubusercontent.com/habibabnk/accidents_project/main/data/2022/2022.csv",
    "2023": "https://raw.githubusercontent.com/habibabnk/accidents_project/main/data/2023/2023.csv",
    "2024": "https://raw.githubusercontent.com/habibabnk/accidents_project/main/data/2024/2024.csv"
}

def load_sample_data():
    """Load sample data for deployment"""
    all_data = []
    
    for year, url in SAMPLE_DATA_URLS.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), sep=';', encoding='utf-8')
                df['year'] = int(year)
                all_data.append(df)
                st.success(f"Loaded {year} data: {len(df)} records")
            else:
                st.warning(f"Could not load {year} data")
        except Exception as e:
            st.error(f"Error loading {year} data: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        # Create minimal sample data if no URLs work
        st.warning("Using minimal sample data - please upload your data files")
        return create_minimal_sample_data()

def create_minimal_sample_data():
    """Create minimal sample data for demonstration"""
    import numpy as np
    
    # Generate sample data for demonstration
    years = [2021, 2022, 2023, 2024]
    departments = [f"{i:02d}" for i in range(1, 96)]  # French departments
    
    data = []
    for year in years:
        for _ in range(1000):  # 1000 accidents per year
            data.append({
                'year': year,
                'month': np.random.randint(1, 13),
                'day': np.random.randint(1, 29),
                'hour': np.random.randint(0, 24),
                'department': np.random.choice(departments),
                'lum': np.random.randint(1, 6),
                'atm': np.random.randint(1, 10),
                'col': np.random.randint(1, 8),
                'int': np.random.randint(1, 10),
                'severity': np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.02]),  # Most non-fatal
                'fatalities': np.random.choice([0, 1, 2, 3], p=[0.85, 0.12, 0.025, 0.005]),
                'serious_injuries': np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02])
            })
    
    df = pd.DataFrame(data)
    st.info(f"Created sample dataset with {len(df)} records for demonstration")
    return df

def check_local_data():
    """Check if local data files exist"""
    data_dirs = ['2021', '2022', '2023', '2024']
    has_local_data = False
    
    for year_dir in data_dirs:
        if os.path.exists(year_dir):
            csv_files = [f for f in os.listdir(year_dir) if f.endswith('.csv')]
            if csv_files:
                has_local_data = True
                break
    
    return has_local_data
