# app/processing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler 
import os

# Setting to make numbers easier to read on display
pd.options.display.float_format = '{:20.2f}'.format

# Show all columns on output
pd.set_option('display.max_columns', 999)

def load_data(filepath):
    df = pd.read_excel(filepath, sheet_name=0)
    return df

def clean_data(df):
    cleaned_df = df.copy()
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("str")
    
    mask = (
        cleaned_df["Invoice"].str.match("^\\d{6}$") == True
    )
    cleaned_df = cleaned_df[mask]
    
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")
    
    mask = (
        (cleaned_df["StockCode"].str.match("^\\d{5}$") == True)
        | (cleaned_df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True)
        | (cleaned_df["StockCode"].str.match("^PADS$") == True)
    )
    cleaned_df = cleaned_df[mask]
    
    cleaned_df.dropna(subset=["Customer ID"], inplace=True)
    cleaned_df = cleaned_df[cleaned_df["Price"] > 0.0]
    
    cleaned_df["SalesLineTotal"] = cleaned_df["Quantity"] * cleaned_df["Price"]
    
    return cleaned_df

def aggregate_data(cleaned_df):
    aggregated_df = cleaned_df.groupby(by="Customer ID", as_index=False) \
        .agg(
            MonetaryValue=("SalesLineTotal", "sum"),
            Frequency=("Invoice", "nunique"),
            LastInvoiceDate=("InvoiceDate", "max")
        )
    
    max_invoice_date = aggregated_df["LastInvoiceDate"].max()
    aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days
    
    return aggregated_df

def remove_outliers(aggregated_df):
    # Monetary Outliers
    M_Q1 = aggregated_df["MonetaryValue"].quantile(0.25)
    M_Q3 = aggregated_df["MonetaryValue"].quantile(0.75)
    M_IQR = M_Q3 - M_Q1
    monetary_outliers_df = aggregated_df[(aggregated_df["MonetaryValue"] > (M_Q3 + 1.5 * M_IQR)) | (aggregated_df["MonetaryValue"] < (M_Q1 - 1.5 * M_IQR))].copy()
    
    # Frequency Outliers
    F_Q1 = aggregated_df['Frequency'].quantile(0.25)
    F_Q3 = aggregated_df['Frequency'].quantile(0.75)
    F_IQR = F_Q3 - F_Q1
    frequency_outliers_df = aggregated_df[(aggregated_df['Frequency'] > (F_Q3 + 1.5 * F_IQR)) | (aggregated_df['Frequency'] < (F_Q1 - 1.5 * F_IQR))].copy()
    
    # Non-outliers
    non_outliers_df = aggregated_df[~aggregated_df.index.isin(monetary_outliers_df.index) & ~aggregated_df.index.isin(frequency_outliers_df.index)]
    
    return non_outliers_df, monetary_outliers_df, frequency_outliers_df

def scale_data(non_outliers_df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])
    scaled_data_df = pd.DataFrame(scaled_data, index=non_outliers_df.index, columns=("MonetaryValue", "Frequency", "Recency"))
    return scaled_data_df

def perform_clustering(scaled_data_df, max_k=12):
    inertia = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)
        cluster_labels = kmeans.fit_predict(scaled_data_df)
        sil_score = silhouette_score(scaled_data_df, cluster_labels)
        silhouette_scores.append(sil_score)
        inertia.append(kmeans.inertia_)
    
    return k_values, inertia, silhouette_scores

def assign_clusters(scaled_data_df, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000)
    cluster_labels = kmeans.fit_predict(scaled_data_df)
    return cluster_labels

def map_cluster_labels(non_outliers_df, cluster_labels):
    non_outliers_df = non_outliers_df.copy()
    non_outliers_df["Cluster"] = cluster_labels
    return non_outliers_df

def generate_visualization(data, plot_type, filename):
    plt.figure(figsize=(10,6))
    if plot_type == 'hist':
        plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    elif plot_type == 'box':
        sns.boxplot(data=data, color='skyblue')
    plt.savefig(os.path.join('app', 'static', 'images', filename))
    plt.close()

    # app/processing.py

def generate_histogram(data, column, filename):
    plt.figure(figsize=(10,6))
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.savefig(os.path.join('app', 'static', 'images', filename))
    plt.close()

def generate_boxplot(data, column, filename):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=data, color='skyblue')
    plt.title(f'{column} Boxplot')
    plt.xlabel(column)
    plt.savefig(os.path.join('app', 'static', 'images', filename))
    plt.close()

# Similarly, add functions for other plots as needed