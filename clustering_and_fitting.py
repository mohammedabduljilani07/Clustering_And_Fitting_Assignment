"""
Clustering and Fitting Assignment
Student Name: MOHAMMED ABDUL JILANI
Student Number: 24168848

This file performs clustering and fitting analysis on customer segmentation data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit


def plot_relational_plot(df):
    """
    Creates a scatter plot showing the relationship between two variables.
    Plots Annual Income vs Spending Score colored by Gender.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=144)
    
    colors = {'Male': '#3498db', 'Female': '#e74c3c'}
    for gender in df['Gender'].unique():
        mask = df['Gender'] == gender
        ax.scatter(df[mask]['Annual Income (k$)'], 
                  df[mask]['Spending Score (1-100)'],
                  c=colors[gender], label=gender, alpha=0.6, s=60, 
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    ax.set_title('Customer Income vs Spending Behavior', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=144, bbox_inches='tight')
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Creates a histogram showing distribution of a categorical variable.
    Displays Age distribution separated by Gender.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=144)
    
    males = df[df['Gender'] == 'Male']['Age']
    females = df[df['Gender'] == 'Female']['Age']
    
    bins = np.arange(15, 75, 5)
    ax.hist([males, females], bins=bins, label=['Male', 'Female'], 
            alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Customer Age Distribution by Gender', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=144, bbox_inches='tight')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Creates a statistical visualization showing correlations.
    Displays a correlation heatmap of numerical features.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=144)
    
    # Convert Gender to numeric for correlation
    df_numeric = df.copy()
    df_numeric['Gender_Numeric'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Select numerical columns
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Numeric']
    correlation_matrix = df_numeric[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1, mask=mask)
    
    ax.set_title('Correlation Matrix of Customer Features', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=144, bbox_inches='tight')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculates the four statistical moments for a given column.
    
    Parameters:
    df: DataFrame containing the data
    col: Column name to analyze
    
    Returns:
    mean, stddev, skew, excess_kurtosis: The four statistical moments
    """
    mean = np.mean(df[col])
    stddev = np.std(df[col])
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the data and displays exploratory statistics.
    Uses describe(), head(), tail(), and corr() methods.
    """
    print("\n=== Data Preprocessing ===")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nCorrelation Matrix:")
    df_numeric = df.copy()
    df_numeric['Gender_Numeric'] = df['Gender'].map({'Male': 1, 'Female': 0})
    print(df_numeric[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Numeric']].corr())
    
    return df


def writing(moments, col):
    """
    Writes out the statistical moments in a readable format.
    """
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Determine skewness
    if moments[2] < -0.5:
        skew_text = "left"
    elif moments[2] > 0.5:
        skew_text = "right"
    else:
        skew_text = "not"
    
    # Determine kurtosis
    if moments[3] < -0.5:
        kurt_text = "platy"
    elif moments[3] > 0.5:
        kurt_text = "lepto"
    else:
        kurt_text = "meso"
    
    print(f'The data was {skew_text} skewed and {kurt_text}kurtic.')
    return


def perform_clustering(df, col1, col2):
    """
    Performs K-means clustering on two columns.
    Includes elbow method and silhouette score calculation.
    
    Parameters:
    df: DataFrame containing the data
    col1: First column name for clustering
    col2: Second column name for clustering
    
    Returns:
    labels, data, xkmeans, ykmeans, cenlabels: Clustering results
    """
    
    def plot_elbow_method(wcss, best_n):
        """
        Plots the elbow method graph for determining optimal clusters.
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=144)
        ax.plot(range(2, 11), wcss, 'kx-', linewidth=2, markersize=8)
        ax.scatter(best_n, wcss[best_n-2], marker='o', color='red', 
                  facecolors='none', s=100)
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel('WCSS', fontsize=12)
        ax.set_title('Elbow Method', fontsize=14)
        ax.set_xlim(2, 10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('elbow_plot.png', dpi=144, bbox_inches='tight')
        plt.close()
        return
    
    def one_silhouette_inertia(n, xy):
        """ 
        Calculates the silhouette score and WCSS for n clusters.
        """
        kmeans = KMeans(n_clusters=n, n_init=20)
        kmeans.fit(xy)
        labels = kmeans.labels_
        
        _score = silhouette_score(xy, labels)
        _inertia = kmeans.inertia_
        
        return _score, _inertia
    
    # Gather data and scale
    X = df[[col1, col2]].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find best number of clusters
    wcss = []
    best_n, best_score = None, -np.inf
    
    print("\n=== Finding Optimal Clusters ===")
    for n in range(2, 11):
        score, inertia = one_silhouette_inertia(n, X_scaled)
        wcss.append(inertia)
        if score > best_score:
            best_n = n
            best_score = score
        print(f"{n:2g} clusters silhouette score = {score:0.2f}")
    
    print(f"Best number of clusters = {best_n:2g}")
    
    plot_elbow_method(wcss, best_n)
    
    # Perform clustering with best_n
    kmeans = KMeans(n_clusters=best_n, n_init=20)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    # Get cluster centers and back-scale (inverse transform)
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    
    xkmeans = centers_original[:, 0]
    ykmeans = centers_original[:, 1]
    cenlabels = kmeans.predict(centers_scaled)
    
    # Return original (non-scaled) data for plotting
    data = X
    
    print("\n=== Cluster Centers ===")
    for i in range(best_n):
        print(f"Cluster {i}: {col1}={xkmeans[i]:.1f}, {col2}={ykmeans[i]:.1f}, "
              f"Size={np.sum(labels == i)}")
    
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Plots the clustered data with cluster centers marked.
    """
    from matplotlib.colors import ListedColormap
    
    n_clusters = len(np.unique(labels))
    colours = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    cmap = ListedColormap(colours)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=144)
    
    # Plot clusters
    s = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, 
                   marker='o', alpha=0.6, s=60, edgecolors='black', 
                   linewidth=0.5, label='Data')
    
    # Plot cluster centers
    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=cmap, 
              marker='x', s=200, linewidths=3, label='Cluster Centers', 
              edgecolors='black')
    
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    ax.set_title('Customer Segmentation using K-Means Clustering', fontsize=14)
    
    cbar = fig.colorbar(s, ax=ax)
    cbar.set_ticks(np.unique(labels))
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering.png', dpi=144, bbox_inches='tight')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """
    Performs polynomial curve fitting on two columns.
    
    Parameters:
    df: DataFrame containing the data
    col1: Independent variable column name
    col2: Dependent variable column name
    
    Returns:
    data, x, y: Tuple of (original data points, fitted x values, fitted y values)
    """
    
    def polynomial_model(x, a, b, c):
        """Polynomial model for curve fitting"""
        return a * x**2 + b * x + c
    
    # Gather data and prepare for fitting
    X = df[col1].values
    y_data = df[col2].values
    
    # Fit model
    popt, pcov = curve_fit(polynomial_model, X, y_data, p0=[0, 0, 50])
    
    # Calculate uncertainties
    perr = np.sqrt(np.diag(pcov))
    
    print("\n=== Curve Fitting Results ===")
    print(f"Fitted Equation: y = {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.2f}")
    print(f"Parameter uncertainties: a±{perr[0]:.4f}, b±{perr[1]:.4f}, c±{perr[2]:.2f}")
    
    # Predict across x
    x = np.linspace(X.min(), X.max(), 300)
    y = polynomial_model(x, *popt)
    
    # Calculate confidence interval
    residuals = y_data - polynomial_model(X, *popt)
    std_residuals = np.std(residuals)
    confidence_interval = 1.96 * std_residuals
    
    # Store data for plotting
    data = (X, y_data, popt, confidence_interval, std_residuals)
    
    return data, x, y


def plot_fitted_data(data, x, y):
    """
    Plots the fitted curve with original data points.
    """
    X, y_data, popt, confidence_interval, std_residuals = data
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=144)
    
    # Plot original data
    ax.plot(X, y_data, 'bo', alpha=0.5, markersize=5, label='Observed Data')
    
    # Plot fitted curve
    ax.plot(x, y, 'k-', linewidth=2, label='Fitted Polynomial Curve')
    
    # Plot confidence interval
    ax.fill_between(x, y - confidence_interval, y + confidence_interval,
                    alpha=0.2, color='gray', label='95% Confidence Interval')
    
    # Add error bars for sample of points
    sample_indices = np.random.choice(len(X), size=20, replace=False)
    ax.errorbar(X[sample_indices], y_data[sample_indices], 
               yerr=std_residuals, fmt='none', ecolor='gray', 
               alpha=0.3, capsize=2)
    
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    ax.set_title('Polynomial Curve Fitting: Income vs Spending', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fitting.png', dpi=144, bbox_inches='tight')
    plt.close()
    return


def main():
    """
    Main function that orchestrates the entire analysis.
    """
    print("=" * 80)
    print("CLUSTERING AND FITTING ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data.csv')
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess
    df = preprocessing(df)
    
    # Choose column for statistical analysis
    col = 'Age'
    
    # Generate plots
    print("\n=== Generating Plots ===")
    print("Creating relational plot...")
    plot_relational_plot(df)
    
    print("Creating statistical plot...")
    plot_statistical_plot(df)
    
    print("Creating categorical plot...")
    plot_categorical_plot(df)
    
    # Statistical analysis
    moments = statistical_analysis(df, col)
    writing(moments, col)
    
    # Clustering
    clustering_results = perform_clustering(df, 'Annual Income (k$)', 'Spending Score (1-100)')
    plot_clustered_data(*clustering_results)
    
    # Fitting
    fitting_results = perform_fitting(df, 'Annual Income (k$)', 'Spending Score (1-100)')
    plot_fitted_data(*fitting_results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Generated files:")
    print("  - relational_plot.png")
    print("  - categorical_plot.png")
    print("  - statistical_plot.png")
    print("  - elbow_plot.png")
    print("  - clustering.png")
    print("  - fitting.png")
    print("=" * 80)
    
    return


if __name__ == '__main__':
    main()
