import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st

def load_data(file):
    df = pd.read_csv(file, sep='\s+', header=None)
    return df

def process_ecg_signal(df):
    ecg_signal = df[df.columns[0]]

    # Calculate the number of samples
    N = len(ecg_signal)

    # Calculate the elapsed time
    sample_interval = np.arange(0, N)
    elapsed_time = sample_interval * (1/125)

    # Center the ECG signal by dividing by 1e8
    y = ecg_signal / 1e8

    return elapsed_time, y

# Streamlit app


uploaded_file = st.file_uploader('Upload your ECG data file', type=['txt', 'csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    elapsed_time, y = process_ecg_signal(df)

    st.write('## Processed ECG Signal')
    
    # Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=elapsed_time, y=y, mode='lines', name='ECG', line=dict(color='blue')))
    fig.update_layout(
        height=500,
        width=1500,
        title="Plot Data ECG (a)",
        xaxis_title="Elapsed Time (s)",
        yaxis_title="Amplitude",
    )
    st.plotly_chart(fig)
else:
    st.write('Please upload an ECG data file to get started.')