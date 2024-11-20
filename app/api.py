from flask import Blueprint, request, jsonify, send_from_directory
import os
import pandas as pd
from .processing import (
    load_data, clean_data, aggregate_data, remove_outliers,
    scale_data, assign_clusters, map_cluster_labels,
    generate_histogram, generate_boxplot
)

api_bp = Blueprint('api', __name__)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'online_retail_II.xlsx')
IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'static', 'images')

# Ensure the images directory exists
os.makedirs(IMAGES_PATH, exist_ok=True)


@api_bp.route('/process', methods=['GET'])
def process_data():
    try:
        df = load_data(DATA_PATH)
        cleaned_df = clean_data(df)
        aggregated_df = aggregate_data(cleaned_df)
        non_outliers_df, _, _ = remove_outliers(aggregated_df)
        scaled_data_df = scale_data(non_outliers_df)
        cluster_labels = assign_clusters(scaled_data_df, n_clusters=4)
        non_outliers_df = map_cluster_labels(non_outliers_df, cluster_labels)

        return jsonify({
            'status': 'success',
            'data': non_outliers_df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/generate_plots', methods=['GET'])
def generate_plots():
    try:
        df = load_data(DATA_PATH)
        cleaned_df = clean_data(df)
        aggregated_df = aggregate_data(cleaned_df)
        non_outliers_df, _, _ = remove_outliers(aggregated_df)
        
        # Generate plots
        generate_histogram(non_outliers_df['MonetaryValue'], 'Monetary Value', os.path.join(IMAGES_PATH, 'monetary_hist.png'))
        generate_histogram(non_outliers_df['Frequency'], 'Frequency', os.path.join(IMAGES_PATH, 'frequency_hist.png'))
        generate_histogram(non_outliers_df['Recency'], 'Recency', os.path.join(IMAGES_PATH, 'recency_hist.png'))
        generate_boxplot(non_outliers_df['MonetaryValue'], 'Monetary Value', os.path.join(IMAGES_PATH, 'monetary_box.png'))
        generate_boxplot(non_outliers_df['Frequency'], 'Frequency', os.path.join(IMAGES_PATH, 'frequency_box.png'))
        generate_boxplot(non_outliers_df['Recency'], 'Recency', os.path.join(IMAGES_PATH, 'recency_box.png'))
        
        return jsonify({'status': 'success', 'message': 'Plots generated successfully.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/visualization/<plot_type>/<filename>', methods=['GET'])
def get_visualization(plot_type, filename):
    if plot_type not in ['hist', 'box']:
        return jsonify({'status': 'error', 'message': 'Invalid plot type.'}), 400
    return send_from_directory(IMAGES_PATH, filename)
