import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings

# Safe Streamlit imports with fallback
cache_data = None
cache_resource = None
try:
    import streamlit as st
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except Exception:
    st = None
    def cache_data(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    def cache_resource(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

# Import deployment data handler
try:
    from deploy_data import load_sample_data, check_local_data
except ImportError:
    def load_sample_data():
        return pd.DataFrame()
    def check_local_data():
        return False

warnings.filterwarnings('ignore')

class AccidentDataLoader:
    def __init__(self, data_dir="accidents_project"):
        self.data_dir = Path(data_dir)
        self.data = None
        self.merged_data = None
        self.loading_errors = []
        
    def smart_read_csv(self, file_path):
        """Robust CSV reader that tries multiple separators and encodings"""
        separators = [";", ","]
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        
        for sep in separators:
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=enc, low_memory=False)
                    return df
                except Exception as e:
                    last_error = e
                    continue
        
        # If all attempts failed, raise with detailed error
        raise RuntimeError(f"Failed reading {file_path}: {last_error}")
        
    def detect_separator(self, file_path):
        """Detect CSV separator by reading first line"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                if ';' in first_line and ',' not in first_line:
                    return ';'
                elif ',' in first_line and ';' not in first_line:
                    return ','
                else:
                    return ';'  # Default for French data
        except:
            return ';'
    
    def detect_encoding(self, file_path):
        """Try different encodings"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try to read a bit
                return encoding
            except:
                continue
        return 'utf-8'  # Default fallback
    
    def load_csv_safe(self, file_path):
        """Load CSV with robust error handling"""
        try:
            df = self.smart_read_csv(file_path)
            return df
        except Exception as e:
            error_info = {
                'file': str(file_path),
                'error': str(e),
                'size': file_path.stat().st_size if file_path.exists() else 0
            }
            self.loading_errors.append(error_info)
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_yearly_data(self):
        """Load all yearly accident data"""
        all_data = []
        
        # Check if we're in deployment (no local data)
        if not check_local_data():
            print("No local data found, loading sample data for deployment")
            self.data = load_sample_data()
            return self.data
        
        for year in range(2015, 2025):
            year_dir = self.data_dir / str(year)
            
            if not year_dir.exists():
                print(f"Year directory not found: {year_dir}")
                continue
            
            # Try different file patterns
            patterns = [
                f"{year}.csv",
                f"caracteristiques-{year}.csv",
                f"carct-{year}.csv",
                f"caract-{year}.csv"
            ]
            
            year_data = None
            for pattern in patterns:
                file_path = year_dir / pattern
                if file_path.exists():
                    year_data = self.load_csv_safe(file_path)
                    if year_data is not None:
                        print(f"Loaded {year} from {pattern}: {len(year_data)} records")
                        break
            
            if year_data is not None:
                year_data['year'] = year
                all_data.append(year_data)
            else:
                print(f"No data could be loaded for year {year}")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"Total records loaded: {len(self.data)}")
            return self.data
        else:
            # Fallback to sample data if no local data found
            print("No local data found, loading sample data")
            self.data = load_sample_data()
            return self.data
    
    def load_detailed_tables(self):
        """Load detailed tables for recent years (2021-2024)"""
        detailed_data = {}
        
        for year in range(2021, 2025):
            year_dir = self.data_dir / str(year)
            detailed_data[year] = {}
            
            # Table patterns to try
            table_patterns = {
                'caracteristiques': [f'caracteristiques-{year}.csv', f'carct-{year}.csv', f'caract-{year}.csv'],
                'usagers': [f'usagers-{year}.csv'],
                'lieux': [f'lieux-{year}.csv'],
                'vehicules': [f'vehicules-{year}.csv']
            }
            
            for table_name, patterns in table_patterns.items():
                for pattern in patterns:
                    file_path = year_dir / pattern
                    if file_path.exists():
                        df = self.load_csv_safe(file_path)
                        if df is not None:
                            detailed_data[year][table_name] = df
                            print(f"Loaded {year} {table_name}: {len(df)} records")
                            break
        
        return detailed_data
    
    def merge_detailed_data(self, detailed_data):
        """Merge detailed tables by year"""
        merged_by_year = []
        
        for year, tables in detailed_data.items():
            if not tables:
                continue
                
            # Start with characteristics table
            if 'caracteristiques' in tables:
                merged = tables['caracteristiques'].copy()
                
                # Add user severity information
                if 'usagers' in tables:
                    usagers = tables['usagers']
                    # Calculate severity statistics per accident
                    severity_stats = usagers.groupby('Num_Acc').agg({
                        'grav': ['count', lambda x: (x >= 3).sum(), lambda x: (x == 4).sum()]
                    }).round(2)
                    severity_stats.columns = ['total_usagers', 'serious_injuries', 'fatalities']
                    severity_stats = severity_stats.reset_index()
                    
                    merged = merged.merge(severity_stats, on='Num_Acc', how='left')
                
                merged['year'] = year
                merged_by_year.append(merged)
                print(f"Merged {year}: {len(merged)} records")
        
        if merged_by_year:
            self.merged_data = pd.concat(merged_by_year, ignore_index=True)
            print(f"Total merged records: {len(self.merged_data)}")
            return self.merged_data
        else:
            return None
    
    def preprocess_data(self, df):
        """Clean and preprocess data"""
        if df is None:
            return None
            
        df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Num_Acc': 'accident_id',
            'jour': 'day',
            'mois': 'month', 
            'an': 'year',
            'hrmn': 'time',
            'lum': 'lighting',
            'dep': 'department',
            'com': 'commune',
            'agg': 'localization',
            'int': 'intersection',
            'atm': 'weather',
            'col': 'collision_type',
            'adr': 'address',
            'lat': 'latitude',
            'long': 'longitude'
        }
        
        # Apply mapping where columns exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
        # Convert time to hour
        if 'time' in df.columns:
            df['hour'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.hour
            df['hour'] = df['hour'].fillna(df['time'].str[:2].astype(float, errors='ignore'))
        
        # Create datetime
        if all(col in df.columns for col in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
            df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create severity target for modeling
        if 'fatalities' in df.columns:
            df['has_fatalities'] = (df['fatalities'] > 0).astype(int)
        elif 'grav' in df.columns:
            df['has_fatalities'] = (df['grav'] == 4).astype(int)
        else:
            df['has_fatalities'] = 0
        
        # Create serious accident target
        if 'serious_injuries' in df.columns and 'fatalities' in df.columns:
            df['is_serious'] = ((df['serious_injuries'] > 0) | (df['fatalities'] > 0)).astype(int)
        elif 'grav' in df.columns:
            df['is_serious'] = (df['grav'] >= 3).astype(int)
        else:
            df['is_serious'] = 0
        
        return df
    
    def get_data(self, data_dir=None):
        """Get processed data with explicit directory support"""
        if data_dir:
            self.data_dir = Path(data_dir)
            self.data = None  # Reset cached data
            self.merged_data = None
            self.loading_errors = []
        
        if self.data is None:
            self.load_yearly_data()
        
        # Try to load detailed data for recent years
        try:
            detailed_data = self.load_detailed_tables()
            if detailed_data:
                merged_data = self.merge_detailed_data(detailed_data)
                if merged_data is not None:
                    processed_merged = self.preprocess_data(merged_data)
                    return processed_merged, self.loading_errors
        except Exception as e:
            print(f"Warning: Failed to load detailed tables: {e}")
            print("Falling back to basic yearly data...")
        
        # Fallback to basic yearly data
        processed_basic = self.preprocess_data(self.data)
        return processed_basic, self.loading_errors
    
    def get_feature_columns(self, df):
        """Get columns suitable for modeling"""
        if df is None:
            return []
        
        categorical_features = [
            'month', 'day_of_week', 'hour', 'department', 'lighting', 
            'weather', 'intersection', 'collision_type', 'localization'
        ]
        
        numerical_features = [
            'year'
        ]
        
        available_features = []
        for feature in categorical_features + numerical_features:
            if feature in df.columns:
                available_features.append(feature)
        
        return available_features, categorical_features, numerical_features

# Global data loader instance
_data_loader = None

def get_data_loader():
    """Get or create data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = AccidentDataLoader()
    return _data_loader

@cache_data
def load_accident_data(data_dir=None):
    """Streamlit cached data loading function with directory support"""
    loader = get_data_loader()
    return loader.get_data(data_dir)

def debug_data_directory(data_dir):
    """Debug function to scan and list all CSV files"""
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            return 0, [f"Directory does not exist: {data_path}"]
        
        # Recursively find all CSV files
        csv_files = list(data_path.rglob("*.csv"))
        total_count = len(csv_files)
        
        # Create preview list (first 30 files)
        file_preview = []
        for file_path in csv_files[:30]:
            relative_path = file_path.relative_to(data_path)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_preview.append(f"{relative_path} ({size_mb:.1f} MB)")
        
        return total_count, file_preview
        
    except Exception as e:
        return 0, [f"Error scanning directory: {e}"]
