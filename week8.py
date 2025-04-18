# iris_analysis.py
"""
Iris Dataset Analysis Project
Run this in VS Code with a Python interpreter
"""

# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_data():
    """Load and prepare the Iris dataset"""
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """Perform initial data exploration"""
    print("\n=== Data Exploration ===")
    
    # Display first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Dataset information
    print("\nDataset info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

def clean_data(df):
    """Handle missing values if any exist"""
    if df.isnull().sum().sum() > 0:
        print("\nCleaning data...")
        df_clean = df.dropna()  # or df.fillna()
        return df_clean
    return df

# Task 2: Basic Data Analysis
def analyze_data(df):
    """Perform statistical analysis"""
    print("\n=== Data Analysis ===")
    
    # Basic statistics
    print("\nDescriptive statistics:")
    print(df.describe())
    
    # Group by species
    print("\nMean by species:")
    print(df.groupby('species').mean())
    
    # Interesting finding
    print("\nPetal length comparison:")
    print(df.groupby('species')['petal length (cm)'].agg(['mean', 'std']))

# Task 3: Data Visualization
def visualize_data(df):
    """Create visualizations"""
    print("\n=== Data Visualization ===")
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 1. Line chart (simulated time series)
    plt.subplot(2, 2, 1)
    df['sepal length (cm)'].plot(kind='line')
    plt.title('Sepal Length Trend')
    plt.ylabel('Length (cm)')
    
    # 2. Bar chart
    plt.subplot(2, 2, 2)
    df.groupby('species')['petal length (cm)'].mean().plot(kind='bar')
    plt.title('Avg Petal Length by Species')
    plt.ylabel('Length (cm)')
    
    # 3. Histogram
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='sepal width (cm)', bins=15, kde=True)
    plt.title('Sepal Width Distribution')
    
    # 4. Scatter plot
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
    plt.title('Sepal vs Petal Length')
    
    plt.tight_layout()
    plt.show()
    
    # Bonus: Pairplot
    sns.pairplot(df, hue='species')
    plt.suptitle('Feature Relationships', y=1.02)
    plt.show()

def main():
    """Main execution function"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Explore data
    explore_data(df)
    
    # Clean data (though Iris has no missing values)
    df = clean_data(df)
    
    # Analyze data
    analyze_data(df)
    
    # Visualize data
    visualize_data(df)

if __name__ == "__main__":
    main()