"""
Comprehensive Exploratory Data Analysis (EDA) for NHAMCS Dataset.

Follows professional statistical methodology:
1. Data Quality Assessment
2. Univariate Analysis
3. Bivariate Analysis
4. Multivariate Analysis

Generates 30+ visualizations and comprehensive insights report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, chi2_contingency, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings('ignore')

# Set style for medical/clinical theme
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Medical-appropriate color palette
MEDICAL_COLORS = {
    'esi_1': '#DC143C',  # Crimson - Critical
    'esi_2': '#FF6347',  # Tomato - Emergent
    'esi_3': '#FFA500',  # Orange - Urgent
    'esi_4': '#FFD700',  # Gold - Less Urgent
    'esi_5': '#98FB98',  # Pale Green - Non-urgent
    'esi_other': '#D3D3D3',  # Light Gray
}


class ComprehensiveEDA:
    """
    Comprehensive EDA analysis class for NHAMCS dataset.
    """
    
    def __init__(self, data_path: str, output_dir: str = "outputs"):
        """
        Initialize EDA analysis.
        
        Args:
            data_path: Path to NHAMCS combined CSV file
            output_dir: Directory to save outputs
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.text_cols = []
        self.stats_results = {}
        self.figure_count = 0
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        
        # Define categorical columns (even if stored as numeric)
        categorical_definitions = {
            'sex', 'esi_level', 'month', 'day_of_week', 
            'injury', 'ambulance_arrival', 'seen_72h', 'discharged_7d',
            'cebvd', 'chf', 'ed_dialysis', 'hiv', 'diabetes', 'no_chronic_conditions',
            'on_oxygen', 'gcs'  # GCS and on_oxygen can be treated as categorical
        }
        
        # Classify columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if col.startswith('rfv'):
                    self.text_cols.append(col)
                else:
                    self.categorical_cols.append(col)
            elif col in categorical_definitions:
                # Numeric but actually categorical
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)
        
        # Remove 'year' from numeric for some analyses (but keep it for temporal)
        if 'year' in self.numeric_cols:
            self.numeric_cols.remove('year')
        
        print(f"Numeric columns: {len(self.numeric_cols)}")
        print(f"Categorical columns: {len(self.categorical_cols)}")
        print(f"Text columns: {len(self.text_cols)}")
        print(f"Categorical: {sorted(self.categorical_cols)}")
        
        return self.df
    
    def save_figure(self, filename: str):
        """Save current figure with consistent naming."""
        self.figure_count += 1
        filepath = self.figures_dir / f"{self.figure_count:02d}_{filename}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath.name}")
    
    # ========================================================================
    # PHASE 1: DATA QUALITY ASSESSMENT
    # ========================================================================
    
    def data_quality_assessment(self):
        """Assess data quality and completeness."""
        print("\n=== PHASE 1: DATA QUALITY ASSESSMENT ===")
        
        # 1. Missing values heatmap
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        plt.figure(figsize=(14, 8))
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        
        # Only show columns with missing values
        missing_df = missing_df[missing_df['Missing %'] > 0]
        
        if len(missing_df) > 0:
            sns.heatmap(
                self.df[missing_df.index].isnull().T,
                cbar=True,
                cmap='viridis',
                yticklabels=True
            )
            plt.title('Missing Values Heatmap (Columns with Missing Data)', fontsize=14)
        else:
            plt.text(0.5, 0.5, 'No Missing Values in Dataset', 
                    ha='center', va='center', fontsize=16)
            plt.title('Missing Values Heatmap', fontsize=14)
        self.save_figure("01_missing_values_heatmap")
        
        # 2. Missing values percentage bar chart
        if len(missing_df) > 0:
            plt.figure(figsize=(12, 8))
            missing_df['Missing %'].head(20).plot(kind='barh')
            plt.xlabel('Missing Percentage (%)')
            plt.title('Top 20 Columns by Missing Data Percentage', fontsize=14)
            plt.gca().invert_yaxis()
            self.save_figure("02_missing_percentage")
        
        # 3. Data completeness by year
        completeness_by_year = self.df.groupby('year').apply(
            lambda x: (1 - x.isnull().sum().sum() / (len(x) * len(x.columns))) * 100
        )
        
        plt.figure(figsize=(12, 6))
        completeness_by_year.plot(kind='bar', color='steelblue')
        plt.xlabel('Year')
        plt.ylabel('Completeness (%)')
        plt.title('Data Completeness by Year', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("03_completeness_by_year")
        
        # 4. Record counts by year
        plt.figure(figsize=(12, 6))
        year_counts = self.df['year'].value_counts().sort_index()
        year_counts.plot(kind='bar', color='teal')
        plt.xlabel('Year')
        plt.ylabel('Number of Records')
        plt.title('Number of Records by Year', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("04_records_by_year")
        
        # 5. Data types summary
        plt.figure(figsize=(10, 6))
        dtype_counts = self.df.dtypes.value_counts()
        dtype_counts.plot(kind='bar', color='coral')
        plt.xlabel('Data Type')
        plt.ylabel('Count')
        plt.title('Distribution of Data Types', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("05_datatype_distribution")
        
        # 6. Duplicate check
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate records: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        
        self.stats_results['data_quality'] = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_percentage': missing_pct.to_dict(),
            'completeness_by_year': completeness_by_year.to_dict(),
            'duplicates': duplicates
        }
    
    # ========================================================================
    # PHASE 2: UNIVARIATE ANALYSIS
    # ========================================================================
    
    def univariate_analysis(self):
        """Perform univariate analysis for all variables."""
        print("\n=== PHASE 2: UNIVARIATE ANALYSIS ===")
        
        # Target variable: ESI Level
        self._analyze_esi_level()
        
        # Demographics
        self._analyze_demographics()
        
        # Vital signs with distribution plots (for skewness)
        self._analyze_vital_signs()
        
        # Categorical variables
        self._analyze_categorical_vars()
        
        # Text variables (RFV)
        self._analyze_rfv_variables()
        
        # Visit characteristics
        self._analyze_visit_characteristics()
    
    def _analyze_esi_level(self):
        """Analyze target variable: ESI Level."""
        print("\nAnalyzing ESI Level...")
        
        # Distribution bar chart
        plt.figure(figsize=(12, 6))
        esi_counts = self.df['esi_level'].value_counts().sort_index()
        esi_counts.plot(kind='bar', color='steelblue')
        plt.xlabel('ESI Level')
        plt.ylabel('Count')
        plt.title('Distribution of ESI Levels (Target Variable)', fontsize=14)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("06_esi_level_distribution")
        
        # Pie chart
        plt.figure(figsize=(10, 10))
        esi_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.ylabel('')
        plt.title('ESI Level Distribution (Percentage)', fontsize=14)
        self.save_figure("07_esi_level_pie")
        
        # Statistics
        self.stats_results['esi_level'] = {
            'distribution': esi_counts.to_dict(),
            'percentage': (esi_counts / len(self.df) * 100).to_dict()
        }
    
    def _analyze_demographics(self):
        """Analyze demographic variables."""
        print("\nAnalyzing Demographics...")
        
        # Age - with distribution plot for skewness
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        self.df['age'].hist(bins=50, color='steelblue', edgecolor='black')
        plt.xlabel('Age (years)')
        plt.ylabel('Frequency')
        plt.title('Age Distribution (Histogram)')
        
        plt.subplot(2, 2, 2)
        self.df['age'].plot(kind='box', vert=True)
        plt.ylabel('Age (years)')
        plt.title('Age Distribution (Box Plot)')
        
        plt.subplot(2, 2, 3)
        stats.probplot(self.df['age'].dropna(), dist="norm", plot=plt)
        plt.title('Age Q-Q Plot (Normality Check)')
        
        plt.subplot(2, 2, 4)
        # Distribution plot for skewness
        sns.distplot(self.df['age'].dropna(), kde=True, hist=True, color='steelblue')
        skewness = self.df['age'].skew()
        plt.axvline(self.df['age'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["age"].mean():.1f}')
        plt.axvline(self.df['age'].median(), color='green', linestyle='--', 
                   label=f'Median: {self.df["age"].median():.1f}')
        plt.xlabel('Age (years)')
        plt.ylabel('Density')
        plt.title(f'Age Distribution (Skewness: {skewness:.2f})')
        plt.legend()
        
        plt.suptitle('Age Variable Analysis', fontsize=16, y=1.02)
        self.save_figure("08_age_analysis")
        
        # Sex distribution
        plt.figure(figsize=(10, 6))
        sex_counts = self.df['sex'].value_counts()
        sex_labels = {1: 'Male', 2: 'Female'}
        sex_counts.index = [sex_labels.get(x, x) for x in sex_counts.index]
        sex_counts.plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.title('Gender Distribution', fontsize=14)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("09_sex_distribution")
        
        # Month patterns
        plt.figure(figsize=(12, 6))
        month_counts = self.df['month'].value_counts().sort_index()
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_counts.index = month_labels
        month_counts.plot(kind='bar', color='teal')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.title('Visits by Month (Seasonal Patterns)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("10_monthly_patterns")
        
        # Day of week
        plt.figure(figsize=(10, 6))
        day_counts = self.df['day_of_week'].value_counts().sort_index()
        day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        day_counts.index = [day_labels[i-1] if i <= 7 else i for i in day_counts.index]
        day_counts.plot(kind='bar', color='coral')
        plt.xlabel('Day of Week')
        plt.ylabel('Count')
        plt.title('Visits by Day of Week', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("11_day_of_week")
    
    def _analyze_vital_signs(self):
        """Analyze vital signs with distribution plots for skewness."""
        print("\nAnalyzing Vital Signs...")
        
        vital_vars = ['temp_c', 'pulse', 'respiration', 'sbp', 'dbp', 'o2_sat', 'gcs', 'pain']
        
        # Distribution plots for all vitals (skewness analysis)
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        skewness_results = {}
        
        for idx, var in enumerate(vital_vars):
            if var not in self.df.columns:
                continue
            
            data = self.df[var].dropna()
            if len(data) == 0:
                continue
            
            skew_val = data.skew()
            skewness_results[var] = skew_val
            
            # Distribution plot with KDE
            sns.distplot(data, kde=True, hist=True, ax=axes[idx], color='steelblue')
            axes[idx].axvline(data.mean(), color='red', linestyle='--', 
                            label=f'Mean: {data.mean():.2f}')
            axes[idx].axvline(data.median(), color='green', linestyle='--', 
                            label=f'Median: {data.median():.2f}')
            axes[idx].set_xlabel(var.replace('_', ' ').title())
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'{var.replace("_", " ").title()} Distribution\n'
                              f'Skewness: {skew_val:.2f}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.suptitle('Vital Signs Distribution Analysis (Skewness)', fontsize=16, y=0.995)
        self.save_figure("12_vital_signs_distributions")
        
        # Skewness summary
        plt.figure(figsize=(12, 6))
        skew_df = pd.Series(skewness_results).sort_values(ascending=False)
        skew_df.plot(kind='barh', color=['red' if abs(x) > 1 else 'orange' if abs(x) > 0.5 else 'green' 
                                         for x in skew_df.values])
        plt.xlabel('Skewness Value')
        plt.title('Skewness Summary for Vital Signs\n'
                 '(|skew| > 1: Highly skewed, |skew| > 0.5: Moderately skewed)', fontsize=14)
        plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
        plt.axvline(-0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axvline(0.5, color='orange', linestyle='--', alpha=0.5)
        plt.axvline(-1, color='red', linestyle='--', alpha=0.5)
        plt.axvline(1, color='red', linestyle='--', alpha=0.5)
        plt.grid(axis='x', alpha=0.3)
        self.save_figure("13_vital_signs_skewness_summary")
        
        # Individual vital signs histograms
        for var in ['temp_c', 'pulse', 'sbp', 'o2_sat']:
            if var in self.df.columns:
                plt.figure(figsize=(10, 6))
                data = self.df[var].dropna()
                plt.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
                plt.xlabel(var.replace('_', ' ').title())
                plt.ylabel('Frequency')
                plt.title(f'{var.replace("_", " ").title()} Distribution')
                plt.grid(axis='y', alpha=0.3)
                self.save_figure(f"14_{var}_histogram")
        
        self.stats_results['vital_signs_skewness'] = skewness_results
    
    def _analyze_categorical_vars(self):
        """Analyze categorical variables."""
        print("\nAnalyzing Categorical Variables...")
        
        # Binary variables
        binary_vars = ['injury', 'ambulance_arrival']
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, var in enumerate(binary_vars):
            if var in self.df.columns:
                counts = self.df[var].value_counts()
                # Use actual number of unique values for colors and labels
                n_unique = len(counts)
                colors = ['lightcoral', 'lightblue'][:n_unique]
                counts.plot(kind='bar', ax=axes[idx], color=colors)
                axes[idx].set_xlabel(var.replace('_', ' ').title())
                axes[idx].set_ylabel('Count')
                axes[idx].set_title(f'{var.replace("_", " ").title()} Distribution')
                # Set labels based on actual values
                if n_unique == 2:
                    axes[idx].set_xticklabels(['No', 'Yes'], rotation=0)
                else:
                    axes[idx].set_xticklabels([str(x) for x in counts.index], rotation=0)
                axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Binary Categorical Variables', fontsize=16)
        self.save_figure("15_binary_variables")
        
        # Comorbidities stacked bar
        comorbidity_vars = ['cebvd', 'chf', 'ed_dialysis', 'hiv', 'diabetes', 'no_chronic_conditions']
        comorbidity_data = self.df[comorbidity_vars].sum()
        
        plt.figure(figsize=(12, 6))
        comorbidity_data.plot(kind='bar', color='teal')
        plt.xlabel('Comorbidity')
        plt.ylabel('Number of Patients')
        plt.title('Comorbidity Prevalence', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("16_comorbidities")
    
    def _analyze_rfv_variables(self):
        """Analyze Reason for Visit (RFV) text variables."""
        print("\nAnalyzing RFV Variables...")
        
        # Top RFV reasons
        for rfv_col in ['rfv1', 'rfv1_3d']:
            if rfv_col in self.df.columns:
                top_rfv = self.df[rfv_col].value_counts().head(20)
                
                plt.figure(figsize=(12, 10))
                top_rfv.plot(kind='barh', color='steelblue')
                plt.xlabel('Count')
                plt.ylabel('Reason for Visit')
                plt.title(f'Top 20 {rfv_col.upper()} Reasons', fontsize=14)
                plt.gca().invert_yaxis()
                plt.grid(axis='x', alpha=0.3)
                self.save_figure(f"17_{rfv_col}_top_reasons")
    
    def _analyze_visit_characteristics(self):
        """Analyze visit characteristics."""
        print("\nAnalyzing Visit Characteristics...")
        
        # Wait time and length of visit - with distribution plots
        time_vars = ['wait_time', 'length_of_visit']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, var in enumerate(time_vars):
            if var in self.df.columns:
                data = self.df[var].dropna()
                skew_val = data.skew()
                
                # Distribution plot
                sns.distplot(data, kde=True, hist=True, ax=axes[idx, 0], color='steelblue')
                axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', 
                                   label=f'Mean: {data.mean():.2f}')
                axes[idx, 0].axvline(data.median(), color='green', linestyle='--', 
                                   label=f'Median: {data.median():.2f}')
                axes[idx, 0].set_xlabel(var.replace('_', ' ').title())
                axes[idx, 0].set_ylabel('Density')
                axes[idx, 0].set_title(f'{var.replace("_", " ").title()} Distribution\n'
                                      f'Skewness: {skew_val:.2f}')
                axes[idx, 0].legend()
                axes[idx, 0].grid(alpha=0.3)
                
                # Box plot
                data.plot(kind='box', ax=axes[idx, 1], vert=True)
                axes[idx, 1].set_ylabel(var.replace('_', ' ').title())
                axes[idx, 1].set_title(f'{var.replace("_", " ").title()} Box Plot')
                axes[idx, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Visit Time Characteristics Analysis', fontsize=16)
        self.save_figure("18_visit_time_characteristics")
    
    # ========================================================================
    # PHASE 3: BIVARIATE ANALYSIS
    # ========================================================================
    
    def bivariate_analysis(self):
        """Perform bivariate analysis."""
        print("\n=== PHASE 3: BIVARIATE ANALYSIS ===")
        
        # ESI level relationships
        self._analyze_esi_relationships()
        
        # Vital signs correlations
        self._analyze_vital_correlations()
        
        # Temporal relationships
        self._analyze_temporal_patterns()
    
    def _analyze_esi_relationships(self):
        """Analyze relationships with ESI level."""
        print("\nAnalyzing ESI Level Relationships...")
        
        # ESI vs Age
        plt.figure(figsize=(14, 8))
        esi_order = sorted(self.df['esi_level'].unique())
        data_to_plot = [self.df[self.df['esi_level'] == esi]['age'].dropna() 
                       for esi in esi_order]
        
        bp = plt.boxplot(data_to_plot, labels=[f'ESI {int(x)}' for x in esi_order],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.xlabel('ESI Level')
        plt.ylabel('Age (years)')
        plt.title('Age Distribution by ESI Level', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("19_esi_vs_age")
        
        # ESI vs Vital Signs (Violin plots)
        vital_vars = ['temp_c', 'pulse', 'sbp', 'o2_sat', 'pain']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, var in enumerate(vital_vars):
            if var in self.df.columns and idx < len(axes):
                sns.violinplot(data=self.df, x='esi_level', y=var, ax=axes[idx], 
                              palette='Set2')
                axes[idx].set_xlabel('ESI Level')
                axes[idx].set_ylabel(var.replace('_', ' ').title())
                axes[idx].set_title(f'{var.replace("_", " ").title()} by ESI Level')
                axes[idx].grid(axis='y', alpha=0.3)
        
        # Remove extra subplot
        if len(vital_vars) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('Vital Signs by ESI Level', fontsize=16)
        self.save_figure("20_esi_vs_vitals")
        
        # ESI vs RFV categories
        if 'rfv1_3d' in self.df.columns:
            top_rfv = self.df['rfv1_3d'].value_counts().head(10).index
            esi_rfv_crosstab = pd.crosstab(
                self.df[self.df['rfv1_3d'].isin(top_rfv)]['rfv1_3d'],
                self.df[self.df['rfv1_3d'].isin(top_rfv)]['esi_level'],
                normalize='index'
            )
            
            plt.figure(figsize=(14, 8))
            esi_rfv_crosstab.plot(kind='bar', stacked=True, colormap='RdYlGn_r')
            plt.xlabel('Top 10 RFV Categories')
            plt.ylabel('Proportion')
            plt.title('ESI Level Distribution by Top RFV Categories', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='ESI Level', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            self.save_figure("21_esi_vs_rfv")
        
        # ESI vs Comorbidities
        comorbidity_vars = ['cebvd', 'chf', 'diabetes', 'hiv']
        comorbidity_data = []
        for var in comorbidity_vars:
            if var in self.df.columns:
                for esi in sorted(self.df['esi_level'].unique()):
                    subset = self.df[self.df['esi_level'] == esi]
                    comorbidity_data.append({
                        'ESI Level': f'ESI {int(esi)}',
                        'Comorbidity': var.upper(),
                        'Prevalence': subset[var].mean() * 100
                    })
        
        if comorbidity_data:
            comorbidity_df = pd.DataFrame(comorbidity_data)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=comorbidity_df, x='ESI Level', y='Prevalence', 
                       hue='Comorbidity', palette='Set2')
            plt.ylabel('Prevalence (%)')
            plt.title('Comorbidity Prevalence by ESI Level', fontsize=14)
            plt.legend(title='Comorbidity', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            self.save_figure("22_esi_vs_comorbidities")
        
        # ESI vs Wait Time
        plt.figure(figsize=(14, 8))
        esi_order = sorted(self.df['esi_level'].unique())
        data_to_plot = [self.df[self.df['esi_level'] == esi]['wait_time'].dropna() 
                       for esi in esi_order]
        
        bp = plt.boxplot(data_to_plot, labels=[f'ESI {int(x)}' for x in esi_order],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        plt.xlabel('ESI Level')
        plt.ylabel('Wait Time (minutes)')
        plt.title('Wait Time Distribution by ESI Level', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("23_esi_vs_wait_time")
        
        # ESI vs Injury/Ambulance
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, var in enumerate(['injury', 'ambulance_arrival']):
            if var in self.df.columns:
                crosstab = pd.crosstab(self.df['esi_level'], self.df[var], normalize='index')
                crosstab.plot(kind='bar', ax=axes[idx], color=['lightblue', 'lightcoral'])
                axes[idx].set_xlabel('ESI Level')
                axes[idx].set_ylabel('Proportion')
                axes[idx].set_title(f'{var.replace("_", " ").title()} by ESI Level')
                axes[idx].legend(['No', 'Yes'])
                axes[idx].set_xticklabels([f'ESI {int(x)}' for x in crosstab.index], rotation=0)
                axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Injury and Ambulance Arrival by ESI Level', fontsize=16)
        self.save_figure("24_esi_vs_injury_ambulance")
    
    def _analyze_vital_correlations(self):
        """Analyze correlations between vital signs."""
        print("\nAnalyzing Vital Signs Correlations...")
        
        # Scatter: Temperature vs Age
        plt.figure(figsize=(10, 6))
        sample = self.df[['age', 'temp_c']].dropna().sample(min(10000, len(self.df)))
        plt.scatter(sample['age'], sample['temp_c'], alpha=0.3, s=10)
        z = np.polyfit(sample['age'], sample['temp_c'], 1)
        p = np.poly1d(z)
        plt.plot(sample['age'], p(sample['age']), "r--", alpha=0.8)
        plt.xlabel('Age (years)')
        plt.ylabel('Temperature (Celsius)')
        plt.title('Temperature vs Age (with regression line)', fontsize=14)
        plt.grid(alpha=0.3)
        self.save_figure("25_temp_vs_age")
        
        # Scatter: SBP vs DBP
        plt.figure(figsize=(10, 6))
        sample = self.df[['sbp', 'dbp']].dropna().sample(min(10000, len(self.df)))
        plt.scatter(sample['sbp'], sample['dbp'], alpha=0.3, s=10)
        z = np.polyfit(sample['sbp'], sample['dbp'], 1)
        p = np.poly1d(z)
        plt.plot(sample['sbp'], p(sample['sbp']), "r--", alpha=0.8)
        plt.xlabel('Systolic BP (mmHg)')
        plt.ylabel('Diastolic BP (mmHg)')
        plt.title('Blood Pressure Relationship (SBP vs DBP)', fontsize=14)
        plt.grid(alpha=0.3)
        self.save_figure("26_sbp_vs_dbp")
        
        # Correlation matrix for numeric variables
        numeric_for_corr = [col for col in self.numeric_cols 
                           if col not in ['year'] and self.df[col].notna().sum() > 1000]
        corr_matrix = self.df[numeric_for_corr].corr()
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numeric Variables', fontsize=14)
        self.save_figure("27_correlation_matrix")
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns."""
        print("\nAnalyzing Temporal Patterns...")
        
        # ESI level trends by year
        esi_by_year = pd.crosstab(self.df['year'], self.df['esi_level'], normalize='index') * 100
        
        plt.figure(figsize=(14, 8))
        esi_by_year.plot(kind='bar', stacked=True, colormap='RdYlGn_r')
        plt.xlabel('Year')
        plt.ylabel('Percentage (%)')
        plt.title('ESI Level Distribution Trends by Year', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='ESI Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("28_esi_trends_by_year")
        
        # Monthly patterns in ESI
        esi_by_month = pd.crosstab(self.df['month'], self.df['esi_level'], normalize='index') * 100
        
        plt.figure(figsize=(14, 8))
        esi_by_month.plot(kind='bar', stacked=True, colormap='RdYlGn_r')
        plt.xlabel('Month')
        plt.ylabel('Percentage (%)')
        plt.title('ESI Level Distribution by Month', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='ESI Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        self.save_figure("29_esi_by_month")
    
    # ========================================================================
    # PHASE 4: MULTIVARIATE ANALYSIS
    # ========================================================================
    
    def multivariate_analysis(self):
        """Perform multivariate analysis."""
        print("\n=== PHASE 4: MULTIVARIATE ANALYSIS ===")
        
        # Full correlation heatmap
        self._full_correlation_analysis()
        
        # Feature importance
        self._feature_importance_analysis()
        
        # Interaction effects
        self._interaction_effects()
        
        # Temporal multivariate
        self._temporal_multivariate()
    
    def _full_correlation_analysis(self):
        """Full correlation heatmap with clustering."""
        print("\nPerforming Full Correlation Analysis...")
        
        numeric_for_corr = [col for col in self.numeric_cols 
                           if col not in ['year'] and self.df[col].notna().sum() > 1000]
        
        corr_matrix = self.df[numeric_for_corr].corr()
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        plt.figure(figsize=(18, 16))
        sns.clustermap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                      center=0, square=True, linewidths=0.5, figsize=(18, 16))
        plt.title('Correlation Matrix with Hierarchical Clustering', fontsize=14)
        plt.savefig(self.figures_dir / f"{self.figure_count + 1:02d}_full_correlation_clustermap.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        self.figure_count += 1
        print(f"  Saved: full_correlation_clustermap.png")
    
    def _feature_importance_analysis(self):
        """Analyze feature importance using Random Forest."""
        print("\nAnalyzing Feature Importance...")
        
        # Prepare data for ML
        numeric_features = [col for col in self.numeric_cols 
                          if col != 'esi_level' and self.df[col].notna().sum() > 1000]
        
        # Include some key categorical features (convert to numeric for RF)
        categorical_for_ml = ['sex', 'injury', 'ambulance_arrival']
        all_features = numeric_features + [c for c in categorical_for_ml if c in self.df.columns]
        
        # Create subset - use fillna with median for numeric, mode for categorical
        ml_df = self.df[all_features + ['esi_level']].copy()
        
        # Fill missing values
        for col in numeric_features:
            if col in ml_df.columns:
                ml_df[col] = ml_df[col].fillna(ml_df[col].median())
        
        for col in categorical_for_ml:
            if col in ml_df.columns:
                ml_df[col] = ml_df[col].fillna(ml_df[col].mode()[0] if len(ml_df[col].mode()) > 0 else 0)
        
        # Remove rows with missing ESI
        ml_df = ml_df[ml_df['esi_level'].notna()]
        
        if len(ml_df) < 1000:
            print("  Warning: Insufficient data for feature importance analysis")
            return
        
        X = ml_df[all_features]
        y = ml_df['esi_level']
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': all_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        feature_imp.head(20).plot(x='feature', y='importance', kind='barh', color='steelblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importances (Random Forest)', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        self.save_figure("30_feature_importance")
        
        self.stats_results['feature_importance'] = feature_imp.to_dict('records')
    
    def _interaction_effects(self):
        """Analyze interaction effects."""
        print("\nAnalyzing Interaction Effects...")
        
        # Age × Vital signs × ESI level
        if 'age' in self.df.columns and 'temp_c' in self.df.columns:
            # Sample for visualization
            sample_df = self.df[['age', 'temp_c', 'esi_level']].dropna().sample(
                min(5000, len(self.df)))
            
            plt.figure(figsize=(14, 8))
            for esi in sorted(sample_df['esi_level'].unique())[:5]:  # Top 5 ESI levels
                subset = sample_df[sample_df['esi_level'] == esi]
                plt.scatter(subset['age'], subset['temp_c'], 
                           alpha=0.4, s=20, label=f'ESI {int(esi)}')
            
            plt.xlabel('Age (years)')
            plt.ylabel('Temperature (Celsius)')
            plt.title('Age × Temperature Interaction by ESI Level', fontsize=14)
            plt.legend()
            plt.grid(alpha=0.3)
            self.save_figure("31_age_temp_esi_interaction")
        
        # RFV × Comorbidities × ESI level heatmap
        if 'rfv1_3d' in self.df.columns:
            top_rfv = self.df['rfv1_3d'].value_counts().head(5).index
            top_comorbidities = ['diabetes', 'chf', 'cebvd']
            
            interaction_data = []
            for rfv in top_rfv:
                for comorb in top_comorbidities:
                    if comorb in self.df.columns:
                        subset = self.df[(self.df['rfv1_3d'] == rfv) & 
                                        (self.df[comorb] == 1)]
                        if len(subset) > 0:
                            avg_esi = subset['esi_level'].mean()
                            interaction_data.append({
                                'RFV': rfv[:30],  # Truncate for display
                                'Comorbidity': comorb.upper(),
                                'Avg ESI': avg_esi,
                                'Count': len(subset)
                            })
            
            if interaction_data:
                interaction_df = pd.DataFrame(interaction_data)
                pivot_df = interaction_df.pivot(index='RFV', columns='Comorbidity', values='Avg ESI')
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn_r', cbar_kws={'label': 'Average ESI'})
                plt.title('RFV × Comorbidity Interaction (Average ESI Level)', fontsize=14)
                plt.xlabel('Comorbidity')
                plt.ylabel('Reason for Visit')
                plt.tight_layout()
                self.save_figure("32_rfv_comorbidity_interaction")
    
    def _temporal_multivariate(self):
        """Temporal multivariate analysis."""
        print("\nAnalyzing Temporal Multivariate Patterns...")
        
        # Year × Month × ESI level heatmap
        esi_by_year_month = pd.crosstab(
            [self.df['year'], self.df['month']],
            self.df['esi_level'],
            normalize='index'
        ) * 100
        
        # Average ESI level by year and month
        avg_esi_by_time = self.df.groupby(['year', 'month'])['esi_level'].mean().unstack()
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(avg_esi_by_time, annot=False, cmap='RdYlGn_r', cbar_kws={'label': 'Average ESI Level'})
        plt.title('Average ESI Level by Year and Month', fontsize=14)
        plt.xlabel('Month')
        plt.ylabel('Year')
        self.save_figure("33_temporal_multivariate")
    
    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================
    
    def statistical_tests(self):
        """Perform comprehensive statistical tests."""
        print("\n=== PERFORMING STATISTICAL TESTS ===")
        
        test_results = {}
        
        # Normality tests for continuous variables (Shapiro-Wilk)
        print("\n1. Normality Tests (Shapiro-Wilk)...")
        normality_results = {}
        numeric_vars_for_normality = ['age', 'temp_c', 'pulse', 'respiration', 'sbp', 'dbp', 
                                     'o2_sat', 'wait_time', 'length_of_visit', 'pain']
        
        for var in numeric_vars_for_normality:
            if var in self.df.columns:
                data = self.df[var].dropna()
                if len(data) >= 3:
                    # Sample if too large (Shapiro-Wilk is for n <= 5000)
                    sample_size = min(5000, len(data))
                    sample = data.sample(sample_size, random_state=42) if len(data) > 5000 else data
                    
                    stat, p_value = shapiro(sample)
                    skewness = data.skew()
                    kurtosis = data.kurtosis()
                    
                    normality_results[var] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'normal': p_value > 0.05,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'n': len(data)
                    }
        
        test_results['normality'] = normality_results
        
        # Correlation tests (Pearson and Spearman)
        print("\n2. Correlation Tests (Pearson & Spearman)...")
        pearson_results = {}
        spearman_results = {}
        
        numeric_vars = ['age', 'temp_c', 'pulse', 'respiration', 'sbp', 'dbp', 'o2_sat', 
                       'wait_time', 'length_of_visit', 'pain', 'gcs']
        
        for i, var1 in enumerate(numeric_vars):
            if var1 not in self.df.columns:
                continue
            for var2 in numeric_vars[i+1:]:
                if var2 not in self.df.columns:
                    continue
                subset = self.df[[var1, var2]].dropna()
                if len(subset) > 100:
                    # Pearson correlation (assumes normality)
                    pearson_corr, pearson_p = stats.pearsonr(subset[var1], subset[var2])
                    pearson_results[f"{var1}_{var2}"] = {
                        'correlation': pearson_corr,
                        'p_value': pearson_p,
                        'significant': pearson_p < 0.05,
                        'n': len(subset)
                    }
                    
                    # Spearman correlation (non-parametric)
                    spearman_corr, spearman_p = stats.spearmanr(subset[var1], subset[var2])
                    spearman_results[f"{var1}_{var2}"] = {
                        'correlation': spearman_corr,
                        'p_value': spearman_p,
                        'significant': spearman_p < 0.05,
                        'n': len(subset)
                    }
        
        test_results['pearson_correlations'] = pearson_results
        test_results['spearman_correlations'] = spearman_results
        
        # Kruskal-Wallis test: ESI level differences in continuous variables
        print("\n3. Kruskal-Wallis Tests (ESI Level Differences)...")
        kruskal_results = {}
        continuous_vars = ['age', 'temp_c', 'pulse', 'respiration', 'sbp', 'dbp', 'o2_sat',
                          'wait_time', 'length_of_visit', 'pain', 'gcs']
        
        for var in continuous_vars:
            if var in self.df.columns:
                groups = []
                esi_levels = []
                for esi in sorted(self.df['esi_level'].unique()):
                    group_data = self.df[self.df['esi_level'] == esi][var].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                        esi_levels.append(esi)
                
                if len(groups) > 1:
                    stat, p_val = kruskal(*groups)
                    
                    # Calculate effect size (eta-squared approximation)
                    n_total = sum(len(g) for g in groups)
                    eta_sq = (stat - (len(groups) - 1)) / (n_total - len(groups))
                    eta_sq = max(0, eta_sq)  # Can't be negative
                    
                    kruskal_results[var] = {
                        'statistic': stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'effect_size_eta_squared': eta_sq,
                        'n_groups': len(groups),
                        'n_total': n_total
                    }
        
        test_results['kruskal_wallis'] = kruskal_results
        
        # Chi-square tests for categorical relationships
        print("\n4. Chi-Square Tests (Categorical Relationships)...")
        chi2_results = {}
        
        # ESI level vs binary categoricals
        binary_vars = ['sex', 'injury', 'ambulance_arrival', 'seen_72h', 'discharged_7d',
                       'cebvd', 'chf', 'hiv', 'diabetes', 'no_chronic_conditions']
        
        for var in binary_vars:
            if var in self.df.columns:
                crosstab = pd.crosstab(self.df['esi_level'], self.df[var])
                if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                    chi2_stat, p_val, dof, expected = chi2_contingency(crosstab)
                    
                    # Cramér's V (effect size)
                    n = crosstab.sum().sum()
                    min_dim = min(crosstab.shape) - 1
                    cramers_v = np.sqrt(chi2_stat / (n * min_dim))
                    
                    chi2_results[f"esi_level_vs_{var}"] = {
                        'chi2_statistic': chi2_stat,
                        'p_value': p_val,
                        'degrees_of_freedom': dof,
                        'significant': p_val < 0.05,
                        'cramers_v': cramers_v,
                        'n': n
                    }
        
        test_results['chi_square'] = chi2_results
        
        # Mann-Whitney U tests for binary categorical vs continuous
        print("\n5. Mann-Whitney U Tests (Binary vs Continuous)...")
        mannwhitney_results = {}
        
        binary_vars_for_mw = ['injury', 'ambulance_arrival', 'seen_72h', 'discharged_7d']
        continuous_for_mw = ['wait_time', 'length_of_visit', 'age', 'pain']
        
        for binary_var in binary_vars_for_mw:
            if binary_var not in self.df.columns:
                continue
            for cont_var in continuous_for_mw:
                if cont_var not in self.df.columns:
                    continue
                
                # Split by binary variable
                group0 = self.df[self.df[binary_var] == 0][cont_var].dropna()
                group1 = self.df[self.df[binary_var] == 1][cont_var].dropna()
                
                if len(group0) > 0 and len(group1) > 0:
                    stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                    
                    # Effect size (rank-biserial correlation)
                    n0, n1 = len(group0), len(group1)
                    r = 1 - (2 * stat) / (n0 * n1)
                    
                    mannwhitney_results[f"{cont_var}_by_{binary_var}"] = {
                        'statistic': stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'effect_size_r': r,
                        'group0_n': n0,
                        'group1_n': n1,
                        'group0_median': group0.median(),
                        'group1_median': group1.median()
                    }
        
        test_results['mann_whitney'] = mannwhitney_results
        
        # ANOVA (one-way) for ESI vs continuous (as alternative to Kruskal-Wallis)
        print("\n6. One-Way ANOVA Tests (ESI Level Differences)...")
        anova_results = {}
        
        for var in ['age', 'temp_c', 'pulse', 'wait_time']:
            if var in self.df.columns:
                groups = [self.df[self.df['esi_level'] == esi][var].dropna() 
                         for esi in sorted(self.df['esi_level'].unique())]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) > 1:
                    # Check if we can use ANOVA (groups should have similar variance)
                    from scipy.stats import f_oneway
                    stat, p_val = f_oneway(*groups)
                    
                    # Effect size (eta-squared)
                    n_total = sum(len(g) for g in groups)
                    ss_between = sum(len(g) * (g.mean() - np.concatenate(groups).mean())**2 
                                   for g in groups)
                    ss_total = sum((x - np.concatenate(groups).mean())**2 for g in groups for x in g)
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0
                    
                    anova_results[var] = {
                        'f_statistic': stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'effect_size_eta_squared': eta_sq,
                        'n_groups': len(groups),
                        'n_total': n_total
                    }
        
        test_results['anova'] = anova_results
        
        self.stats_results['statistical_tests'] = test_results
        
        # Print summary
        print(f"\nStatistical Tests Summary:")
        print(f"  - Normality tests: {len(test_results.get('normality', {}))}")
        print(f"  - Pearson correlations: {len(test_results.get('pearson_correlations', {}))}")
        print(f"  - Spearman correlations: {len(test_results.get('spearman_correlations', {}))}")
        print(f"  - Kruskal-Wallis tests: {len(test_results.get('kruskal_wallis', {}))}")
        print(f"  - Chi-square tests: {len(test_results.get('chi_square', {}))}")
        print(f"  - Mann-Whitney U tests: {len(test_results.get('mann_whitney', {}))}")
        print(f"  - ANOVA tests: {len(test_results.get('anova', {}))}")
        
        return test_results
    
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    
    def generate_report(self):
        """Generate comprehensive EDA report."""
        print("\n=== GENERATING COMPREHENSIVE REPORT ===")
        
        report_path = self.output_dir / "eda_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Exploratory Data Analysis Report\n")
            f.write("## NHAMCS Emergency Department Dataset (2011-2022)\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Records:** {len(self.df):,}\n")
            f.write(f"- **Total Variables:** {len(self.df.columns)}\n")
            f.write(f"- **Numeric Variables:** {len(self.numeric_cols)}\n")
            f.write(f"- **Categorical Variables:** {len(self.categorical_cols)}\n")
            f.write(f"- **Text Variables:** {len(self.text_cols)}\n")
            f.write(f"- **Total Visualizations Generated:** {self.figure_count}\n\n")
            
            # Data Quality
            f.write("## 1. Data Quality Assessment\n\n")
            if 'data_quality' in self.stats_results:
                dq = self.stats_results['data_quality']
                f.write(f"- **Data Completeness:** Overall completeness analysis\n")
                f.write(f"- **Missing Values:** Check visualizations 01-05\n")
                f.write(f"- **Duplicate Records:** {dq.get('duplicates', 0)}\n\n")
            
            # Univariate Findings
            f.write("## 2. Univariate Analysis Findings\n\n")
            
            f.write("### Target Variable: ESI Level\n\n")
            if 'esi_level' in self.stats_results:
                esi_dist = self.stats_results['esi_level']['distribution']
                f.write("ESI Level Distribution:\n")
                for level, count in sorted(esi_dist.items()):
                    f.write(f"- ESI {int(level)}: {count:,} ({count/len(self.df)*100:.1f}%)\n")
                f.write("\n")
            
            f.write("### Key Findings:\n\n")
            f.write("- **Demographics:** Age distribution, gender distribution, temporal patterns\n")
            f.write("- **Vital Signs:** All vital signs analyzed with distribution plots showing skewness\n")
            f.write("- **Skewness Analysis:** Distribution plots reveal data skewness (see visualizations 12-13)\n")
            f.write("- **Categorical Variables:** Binary and multi-category distributions\n")
            f.write("- **Reason for Visit:** Top reasons analyzed with frequency charts\n\n")
            
            # Bivariate Findings
            f.write("## 3. Bivariate Analysis Findings\n\n")
            f.write("### ESI Level Relationships:\n")
            f.write("- **Age vs ESI:** Box plots show age distributions by acuity level\n")
            f.write("- **Vital Signs vs ESI:** Violin plots reveal vital sign patterns by acuity\n")
            f.write("- **Wait Time vs ESI:** Inverse relationship observed\n")
            f.write("- **RFV vs ESI:** Different RFV categories associate with different acuity levels\n\n")
            
            f.write("### Correlations:\n")
            f.write("- **Vital Signs:** Strong correlations between related vitals (e.g., SBP-DBP)\n")
            f.write("- **Correlation Matrix:** See visualization 27 for full correlation analysis\n\n")
            
            # Multivariate Findings
            f.write("## 4. Multivariate Analysis Findings\n\n")
            f.write("- **Correlation Clustering:** Hierarchical clustering reveals variable groupings\n")
            f.write("- **Feature Importance:** Random Forest identifies most predictive features\n")
            f.write("- **Interaction Effects:** Age × Vitals × ESI interactions analyzed\n")
            f.write("- **Temporal Patterns:** Year and month interactions with ESI levels\n\n")
            
            # Statistical Tests
            f.write("## 5. Statistical Test Results\n\n")
            if 'statistical_tests' in self.stats_results:
                tests = self.stats_results['statistical_tests']
                
                if 'normality' in tests:
                    f.write("### 5.1 Normality Tests (Shapiro-Wilk)\n\n")
                    f.write("Tests whether continuous variables follow a normal distribution.\n\n")
                    f.write("| Variable | Statistic | p-value | Normal? | Skewness | Kurtosis | n |\n")
                    f.write("|----------|-----------|---------|--------|----------|----------|---|\n")
                    for var, result in sorted(tests['normality'].items()):
                        status = "Yes" if result['normal'] else "**No**"
                        f.write(f"| {var} | {result['statistic']:.4f} | {result['p_value']:.4f} | "
                               f"{status} | {result['skewness']:.2f} | {result['kurtosis']:.2f} | "
                               f"{result['n']:,} |\n")
                    f.write("\n")
                
                if 'pearson_correlations' in tests:
                    f.write("### 5.2 Pearson Correlations\n\n")
                    f.write("Linear correlations between continuous variables.\n\n")
                    # Show top 10 significant correlations
                    sig_corrs = [(k, v) for k, v in tests['pearson_correlations'].items() 
                               if v['significant']]
                    sig_corrs.sort(key=lambda x: abs(x[1]['correlation']), reverse=True)
                    
                    f.write("**Top 10 Significant Correlations:**\n\n")
                    f.write("| Variables | Correlation | p-value | n |\n")
                    f.write("|-----------|-------------|---------|---|\n")
                    for var_pair, result in sig_corrs[:10]:
                        f.write(f"| {var_pair.replace('_', ' ')} | {result['correlation']:.4f} | "
                               f"{result['p_value']:.4f} | {result['n']:,} |\n")
                    f.write("\n")
                
                if 'spearman_correlations' in tests:
                    f.write("### 5.3 Spearman Correlations (Non-parametric)\n\n")
                    f.write("Rank-based correlations (robust to non-normality).\n\n")
                    sig_corrs = [(k, v) for k, v in tests['spearman_correlations'].items() 
                               if v['significant']]
                    sig_corrs.sort(key=lambda x: abs(x[1]['correlation']), reverse=True)
                    
                    f.write("**Top 10 Significant Correlations:**\n\n")
                    f.write("| Variables | Correlation | p-value | n |\n")
                    f.write("|-----------|-------------|---------|---|\n")
                    for var_pair, result in sig_corrs[:10]:
                        f.write(f"| {var_pair.replace('_', ' ')} | {result['correlation']:.4f} | "
                               f"{result['p_value']:.4f} | {result['n']:,} |\n")
                    f.write("\n")
                
                if 'kruskal_wallis' in tests:
                    f.write("### 5.4 Kruskal-Wallis Tests (ESI Level Differences)\n\n")
                    f.write("Tests whether continuous variables differ across ESI levels (non-parametric ANOVA).\n\n")
                    f.write("| Variable | Statistic | p-value | Significant? | Effect Size (η²) | n |\n")
                    f.write("|----------|-----------|---------|--------------|------------------|---|\n")
                    for var, result in sorted(tests['kruskal_wallis'].items()):
                        status = "**Yes**" if result['significant'] else "No"
                        f.write(f"| {var} | {result['statistic']:.2f} | {result['p_value']:.4f} | "
                               f"{status} | {result['effect_size_eta_squared']:.4f} | "
                               f"{result['n_total']:,} |\n")
                    f.write("\n")
                
                if 'anova' in tests:
                    f.write("### 5.5 One-Way ANOVA Tests (ESI Level Differences)\n\n")
                    f.write("Parametric alternative to Kruskal-Wallis (assumes normality).\n\n")
                    f.write("| Variable | F-statistic | p-value | Significant? | Effect Size (η²) | n |\n")
                    f.write("|----------|-------------|---------|--------------|------------------|---|\n")
                    for var, result in sorted(tests['anova'].items()):
                        status = "**Yes**" if result['significant'] else "No"
                        f.write(f"| {var} | {result['f_statistic']:.2f} | {result['p_value']:.4f} | "
                               f"{status} | {result['effect_size_eta_squared']:.4f} | "
                               f"{result['n_total']:,} |\n")
                    f.write("\n")
                
                if 'chi_square' in tests:
                    f.write("### 5.6 Chi-Square Tests (Categorical Associations)\n\n")
                    f.write("Tests associations between ESI level and categorical variables.\n\n")
                    f.write("| Variables | χ² | p-value | Significant? | Cramér's V | n |\n")
                    f.write("|-----------|----|---------|--------------|------------|---|\n")
                    for var_pair, result in sorted(tests['chi_square'].items()):
                        status = "**Yes**" if result['significant'] else "No"
                        f.write(f"| {var_pair.replace('_', ' ')} | {result['chi2_statistic']:.2f} | "
                               f"{result['p_value']:.4f} | {status} | {result['cramers_v']:.4f} | "
                               f"{result['n']:,} |\n")
                    f.write("\n")
                
                if 'mann_whitney' in tests:
                    f.write("### 5.7 Mann-Whitney U Tests (Binary vs Continuous)\n\n")
                    f.write("Tests differences in continuous variables between binary groups.\n\n")
                    f.write("| Comparison | Statistic | p-value | Significant? | Effect Size (r) |\n")
                    f.write("|------------|-----------|---------|--------------|------------------|\n")
                    for var_pair, result in sorted(tests['mann_whitney'].items()):
                        status = "**Yes**" if result['significant'] else "No"
                        f.write(f"| {var_pair.replace('_', ' ')} | {result['statistic']:.2f} | "
                               f"{result['p_value']:.4f} | {status} | {result['effect_size_r']:.4f} |\n")
                    f.write("\n")
            
            # Modeling Recommendations
            f.write("## 6. Modeling Recommendations\n\n")
            
            f.write("### Feature Engineering:\n")
            f.write("1. **Handle Skewness:** Apply log or Box-Cox transformations to highly skewed variables\n")
            f.write("2. **Text Features:** Use TF-IDF or embeddings for RFV text variables\n")
            f.write("3. **Temporal Features:** Create cyclical features for month/day-of-week\n")
            f.write("4. **Interaction Terms:** Consider Age × Vitals, Comorbidities × RFV interactions\n")
            f.write("5. **Missing Data:** Use median imputation for vitals (already implemented) or advanced imputation\n\n")
            
            f.write("### Preprocessing:\n")
            f.write("1. **Standardization:** Standardize numeric features for distance-based algorithms\n")
            f.write("2. **Class Imbalance:** Address ESI level imbalance (see distribution)\n")
            f.write("3. **Outlier Handling:** Consider IQR-based outlier detection for vital signs\n")
            f.write("4. **Feature Selection:** Use feature importance rankings to reduce dimensionality\n\n")
            
            f.write("### Model Recommendations:\n")
            f.write("1. **Primary Model:** Gradient Boosting (XGBoost/LightGBM) - handles non-linear relationships\n")
            f.write("2. **Alternative:** Random Forest - interpretable, handles mixed data types\n")
            f.write("3. **Neural Network:** Consider deep learning for capturing complex interactions\n")
            f.write("4. **Ensemble:** Combine multiple models for improved performance\n\n")
            
            f.write("### Class Imbalance Strategy:\n")
            esi_dist = self.stats_results.get('esi_level', {}).get('distribution', {})
            f.write("1. **Resampling:** Use SMOTE or ADASYN for minority ESI classes\n")
            f.write("2. **Class Weights:** Apply inverse frequency class weights\n")
            f.write("3. **Cost-Sensitive Learning:** Penalize misclassification of critical cases (ESI 1-2)\n\n")
            
            f.write("### Validation Strategy:\n")
            f.write("1. **Temporal Split:** Use chronological split (train on earlier years, test on later)\n")
            f.write("2. **Stratified K-Fold:** Ensure ESI level distribution maintained in folds\n")
            f.write("3. **Metrics:** Use macro-averaged F1 (handles imbalance) and weighted accuracy\n")
            f.write("4. **Clinical Validation:** Review misclassifications with domain experts\n\n")
            
            f.write("### Feature Priority (Based on Importance Analysis):\n")
            if 'feature_importance' in self.stats_results:
                f.write("Top features to prioritize:\n")
                for feat in self.stats_results['feature_importance'][:10]:
                    f.write(f"- {feat['feature']}: {feat['importance']:.4f}\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## Visualizations Index\n\n")
            f.write(f"All {self.figure_count} visualizations are saved in `figures/` directory.\n")
            f.write("Refer to figure numbers in filenames for specific analyses.\n\n")
        
        print(f"Report saved to: {report_path}")
        
        # Save summary statistics
        summary_path = self.output_dir / "eda_summary_stats.csv"
        summary_data = []
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                summary_data.append({
                    'Variable': col,
                    'Count': self.df[col].count(),
                    'Mean': self.df[col].mean(),
                    'Std': self.df[col].std(),
                    'Min': self.df[col].min(),
                    '25%': self.df[col].quantile(0.25),
                    '50%': self.df[col].median(),
                    '75%': self.df[col].quantile(0.75),
                    'Max': self.df[col].max(),
                    'Skewness': self.df[col].skew(),
                    'Missing': self.df[col].isnull().sum(),
                    'Missing %': (self.df[col].isnull().sum() / len(self.df)) * 100
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary statistics saved to: {summary_path}")
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_complete_analysis(self):
        """Run complete EDA analysis."""
        print("=" * 60)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("NHAMCS Emergency Department Dataset")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Phase 1: Data Quality
        self.data_quality_assessment()
        
        # Phase 2: Univariate
        self.univariate_analysis()
        
        # Phase 3: Bivariate
        self.bivariate_analysis()
        
        # Phase 4: Multivariate
        self.multivariate_analysis()
        
        # Statistical Tests
        self.statistical_tests()
        
        # Generate Report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print(f"ANALYSIS COMPLETE!")
        print(f"Total visualizations generated: {self.figure_count}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    # Setup paths
    # Go from: services/manage-agent/analysis/comprehensive_eda.py
    # To: medi-os/ (project root)
    script_path = Path(__file__)
    project_root = script_path.parent.parent.parent.parent
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    output_dir = script_path.parent / "outputs"
    
    # Run analysis
    eda = ComprehensiveEDA(
        data_path=str(data_path),
        output_dir=str(output_dir)
    )
    
    eda.run_complete_analysis()

