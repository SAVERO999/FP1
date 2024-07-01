import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
import math
import streamlit as st 
from plotly.subplots import make_subplots
import plotly.express as px


df = pd.read_csv('dataecgvannofix.txt', sep='\s+', header=None)
ecg_signal = df[df.columns[0]]

# Calculate the number of samples
N = len(ecg_signal)

# Calculate the elapsed time
sample_interval = np.arange(0, N)
elapsed_time = sample_interval * (1/125)

# Center the ECG signal by subtracting the mean
y = ecg_signal/1e8

def dirac(x):
    if x == 0:
        dirac_delta = 1
    else:
        dirac_delta = 0
    result = dirac_delta
    return result

h = []
g = []
n_list = []
for n in range(-2, 2):
    n_list.append(n)
    temp_h = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
    h.append(temp_h)
    temp_g = -2 * (dirac(n) - dirac(n+1))
    g.append(temp_g)

import numpy as np
Hw = np.zeros(20000)
Gw = np.zeros(20000)
i_list = []
fs =125
for i in range(0,fs + 1):
    i_list.append(i)
    reG = 0
    imG = 0
    reH = 0
    imH = 0
    for k in range(-2, 2):
        reG = reG + g[k + abs(-2)] * np.cos(k * 2 * np.pi * i / fs)
        imG = imG - g[k + abs(-2)] * np.sin(k * 2 * np.pi * i / fs)
        reH = reH + h[k + abs(-2)] * np.cos(k * 2 * np.pi * i / fs)
        imH = imH - h[k + abs(-2)] * np.sin(k * 2 * np.pi * i / fs)
    temp_Hw = np.sqrt((reH*2) + (imH*2))
    temp_Gw = np.sqrt((reG*2) + (imG*2))
    Hw[i] = temp_Hw
    Gw[i] = temp_Gw

i_list = i_list[0:round(fs/2)+1]

Q = np.zeros((9, round(fs/2) + 1))

# Generate the i_list and fill Q with the desired values
i_list = []
for i in range(0, round(fs/2) + 1):
    i_list.append(i)
    Q[1][i] = Gw[i]
    Q[2][i] = Gw[2*i] * Hw[i]
    Q[3][i] = Gw[4*i] * Hw[2*i] * Hw[i]
    Q[4][i] = Gw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[5][i] = Gw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[6][i] = Gw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[7][i] = Gw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]
    Q[8][i] = Gw[128*i] * Hw[64*i] * Hw[32*i] * Hw[16*i] * Hw[8*i] * Hw[4*i] * Hw[2*i] * Hw[i]

traces = []



qj = np.zeros((6, 10000))
k_list = []
j = 1

# Calculations
a = -(round(2*j) + round(2*(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range(a, b):
    k_list.append(k)
    qj[1][k + abs(a)] = -2 * (dirac(k) - dirac(k+1))

k_list = []
j= 2
a = -(round (2*j) + round (2*(j-1)) - 2 )
b=-(1- round(2**(j-1)))+1
for k in range (a,b):
  k_list.append(k)
  qj[2][k+abs(a)] = -1/4* ( dirac(k-1) + 3*dirac(k)  + 2*dirac(k+1)  - 2*dirac(k+2) - 3*dirac(k+3) - dirac(k+4))


k_list = []
j=3
a=-(round(2*j) + round(2*(j-1))-2)
b = - (1 - round(2**(j-1))) + 1
for k in range (a,b):
  k_list.append(k)
  qj[3][k+abs(a)] = -1/32*(dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k)
  + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5)
  - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))

k_list = []
j=4
a=-(round(2*j) + round(2*(j-1))-2)
b = - (1 - round(2**(j-1))) + 1

for k in range (a,b):
  k_list.append(k)
  qj [4][k+abs(a)] = -1/256*(dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac (k-3)
  + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 41*dirac(k+1) + 43*dirac(k+2)
  + 42*dirac(k+3) + 38*dirac(k+4) + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7)
  - 8*dirac(k+8) - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
  - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16) - 21*dirac(k+17)
  - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20) - 3*dirac(k+21) - dirac(k+22))

k_list = []
j=5
a=-(round(2*j) + round(2*(j-1))-2)
b = - (1 - round(2**(j-1))) + 1
for k in range (a,b):
  k_list.append(k)
  qj[5][k+abs(a)] = -1/(512)*(dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
+ 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
+ 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
+ 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
+ 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45 *dirac(k+14)
+ 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac (k+20)
- 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
- 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
- 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
- 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
- 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
- dirac(k+46))

k_list = []
j=5
a=-(round(2*j) + round(2*(j-1))-2)
b = - (1 - round(2**(j-1))) + 1
for k in range (a,b):
  k_list.append(k)
  qj[5][k+abs(a)] = -1/(512)*(dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
+ 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
+ 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
+ 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
+ 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45 *dirac(k+14)
+ 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac (k+20)
- 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
- 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
- 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
- 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
- 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
- dirac(k+46))

T1= round (2**(1-1))-1
T2 = round(2** (2-1)) - 1
T3 = round(2** (3-1)) - 1
T4 = round(2**(4-1)) - 1
T5 = round(2**(5-1))- 1
Delay1= T5-T1
Delay2= T5-T2
Delay3= T5-T3
Delay4= T5-T4
Delay5= T5-T5

ecg=y

min_n = 0 * fs
max_n = 8 * fs 


def process_ecg(min_n, max_n, ecg, g, h):
    w2fm = np.zeros((5, max_n - min_n + 1))
    s2fm = np.zeros((5, max_n - min_n + 1))

    for n in range(min_n, max_n + 1):
        for j in range(1, 6):
            w2fm[j-1, n - min_n] = 0
            s2fm[j-1, n - min_n] = 0
            for k in range(-1, 3):
                index = round(n - 2**(j-1) * k)
                if 0 <= index < len(ecg):  # Ensure the index is within bounds
                    w2fm[j-1, n - min_n] += g[k+1] * ecg[index]  # g[k+1] to match Pascal's array index starting from -1
                    s2fm[j-1, n - min_n] += h[k+1] * ecg[index]  # h[k+1] to match Pascal's array index starting from -1

    return w2fm, s2fm

# Compute w2fm and s2fm
w2fm, s2fm = process_ecg(min_n, max_n, ecg, g, h)

# Prepare data for plotting
n_values = np.arange(min_n, max_n + 1)
w2fm_values = [w2fm[i, :] for i in range(5)]  # Equivalent to w2fm[1,n] to w2fm[5,n] in original code (0-based index)
s2fm_values = [s2fm[i, :] for i in range(5)]  # Equivalent to s2fm[1,n] to s2fm[5,n] in original code (0-based index)

w2fb = np.zeros((6, len(ecg) + T5))


n_list = list(range(len(ecg)))

# Perform calculations
for n in n_list:
    for j in range(1, 6):
        w2fb[1][n + T1] = 0
        w2fb[2][n + T2] = 0
        w2fb[3][n + T3] = 0
        a = -(round(2*j) + round(2*(j - 1)) - 2)
        b = -(1 - round(2**(j - 1)))
        for k in range(a, b + 1):
            index = n - (k + abs(a))
            if 0 <= index < len(ecg):
                w2fb[3][n + T3] += qj[3][k + abs(a)] * ecg[index]

# Create and display plots for each DWT level
figs = []
n = np.arange(1000)

gradien1 = np.zeros(len(ecg))
gradien2 = np.zeros(len(ecg))
gradien3 = np.zeros(len(ecg))

# Define delay
delay = T3

# Compute gradien3
N = len(ecg)
for k in range(delay, N - delay):
    gradien3[k] = w2fb[3][k - delay] - w2fb[3][k + delay]
hasil_QRS = np.zeros(len(elapsed_time))
for i in range(N):
    if (gradien3[i] > 1.8):
        hasil_QRS[i-(T4+1)] = 5
    else:
        hasil_QRS[i-(T4+1)] = 0
        
ptp = 0
waktu = np.zeros(np.size(hasil_QRS))
selisih = np.zeros(np.size(hasil_QRS))

for n in range(np.size(hasil_QRS) - 1):
    if hasil_QRS[n] < hasil_QRS[n + 1]:
        waktu[ptp] = n / fs;
        selisih[ptp] = waktu[ptp] - waktu[ptp - 1]
        ptp += 1

ptp = ptp - 1

j = 0
peak = np.zeros(np.size(hasil_QRS))
for n in range(np.size(hasil_QRS)-1):
    if hasil_QRS[n] == 5 and hasil_QRS[n-1] == 0:
        peak[j] = n
        j += 1

temp = 0
interval = np.zeros(np.size(hasil_QRS))
BPM = np.zeros(np.size(hasil_QRS))

for n in range(ptp):
    interval[n] = (peak[n] - peak[n-1]) * (1/fs)
    BPM[n] = 60 / interval[n]
    temp = temp+BPM[n]
    rata = temp / (n - 1)

bpm_rr = np.zeros(ptp)
for n in range (ptp):
  bpm_rr[n] = 60/selisih[n]
  if bpm_rr [n]>100:
    bpm_rr[n]=rata
n = np. arange(0,ptp,1,dtype=int)

#normalisasi tachogram
bpm_rr_baseline = bpm_rr -22

# Plotting dengan Plotly
n = np.arange(0, ptp, 1, dtype=int)

def fourier_transform(signal):
    N = len(signal)
    fft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            fft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return fft_result

def calculate_frequency(N, sampling_rate):
    return np.arange(N) * sampling_rate / N

sampling_rate = 1  # Example sampling rate

fft_results_dict = {}

# Define a list of colors
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']

# Loop for 7 subsets
for i in range(7):
    start_index = i * 20
    end_index = start_index + 320

    n_subset = n[start_index:end_index]
    bpm_rr_baseline_subset = bpm_rr_baseline[start_index:end_index]

    M = len(bpm_rr_baseline_subset) - 1

    hamming_window = np.zeros(M + 1)
    for j in range(M + 1):
        hamming_window[j] = 0.54 - 0.46 * np.cos(2 * np.pi * j / M)

    bpm_rr_baseline_windowed = bpm_rr_baseline_subset * hamming_window

    fft_result = fourier_transform(bpm_rr_baseline_windowed)
    fft_freq = calculate_frequency(len(bpm_rr_baseline_windowed), sampling_rate)

    half_point = len(fft_freq) // 2
    fft_freq_half = fft_freq[:half_point]
    fft_result_half = fft_result[:half_point]

    # Store fft_result_half in the dictionary
    fft_results_dict[f'fft_result{i+1}'] = fft_result_half
    
min_length = min(len(fft_result) for fft_result in fft_results_dict.values())

# Truncate all FFT results to the minimum length
for key in fft_results_dict:
    fft_results_dict[key] = fft_results_dict[key][:min_length]

# Average the FFT results
FFT_TOTAL = sum(fft_results_dict[key] for key in fft_results_dict) / len(fft_results_dict)
fft_freq_half = fft_freq_half[:min_length]  # Truncate frequency array to match

# Frequency bands
x_vlf = np.linspace(0.003, 0.04, 99)
x_lf = np.linspace(0.04, 0.15, 99)
x_hf = np.linspace(0.15, 0.4, 99)

# Interpolation
def manual_interpolation(x, xp, fp):
    return np.interp(x, xp, fp)

y_vlf = manual_interpolation(x_vlf, fft_freq_half, np.abs(FFT_TOTAL))
y_lf = manual_interpolation(x_lf, fft_freq_half, np.abs(FFT_TOTAL))
y_hf = manual_interpolation(x_hf, fft_freq_half, np.abs(FFT_TOTAL))

def trapezoidal_rule(y, x):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

# Hitung Total Power (TP) menggunakan metode trapesium manual
TP = trapezoidal_rule(np.abs(FFT_TOTAL), fft_freq_half)

# Hitung nilai VLF, LF, dan HF menggunakan metode trapesium manual
VLF = trapezoidal_rule(y_vlf, x_vlf)
LF = trapezoidal_rule(y_lf, x_lf)
HF = trapezoidal_rule(y_hf, x_hf)

tp = VLF + LF + HF
# Hitung LF dan HF yang dinormalisasi
LF_norm = LF / (tp - VLF)
HF_norm = HF / (tp- VLF)
LF_HF = LF / HF



with st.sidebar:
    selected = option_menu("FP", ["Home", "DWT","Zeros Crossing","QRS Detection","Frekuensi Domain"], default_index=0)

if selected == "Home":
   st.title('Project ASN Kelompok 6')
   st.subheader("Anggota kelompok")
   new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Afifah Hasnia Nur Rosita - 5023211007</p>'
   st.markdown(new_title, unsafe_allow_html=True)
   new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Syahdifa Aisyah Qurrata Ayun - 5023211032</p>'
   st.markdown(new_title, unsafe_allow_html=True)
   new_title = '<p style="font-family:Georgia; color: black; font-size: 20px;">Sharfina Nabila Larasati - 5023211055</p>'
   st.markdown(new_title, unsafe_allow_html=True)
  


if selected == "DWT":
   sub_selected = st.sidebar.radio(
        "",
        ["Input Data","Filter Coeffs", "Mallat", "Filter Bank"],
        index=0
    )

   if sub_selected  == 'Input Data': 
          # Plot using Plotly
        fig = go.Figure()
        
        # Add the ECG signal trace
        fig.add_trace(go.Scatter(x=elapsed_time, y=y, mode='lines', name='ECG Signal'))
        
        # Update the layout
        fig.update_layout(
            title='ECG Signal',
            xaxis_title='Elapsed Time (s)',
            yaxis_title='Amplitude',
            width=1000,
            height=400
        )
        
        # Show the plot
        st.plotly_chart(fig)
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elapsed_time[0:1000], y=y[0:1000], mode='lines', name='ECG (a)', line=dict(color='blue')))
        fig.update_layout(
            height=500,
            width=1500,
            title="ECG Signal",
            xaxis_title="Elapsed Time (s)",
            yaxis_title="Nilai",
        
        )
        st.plotly_chart(fig)
       
   if sub_selected  == 'Filter Coeffs':

        fig = go.Figure(data=[go.Bar(x=n_list, y=h)])
        fig.update_layout(title='h(n) Plot', xaxis_title='n', yaxis_title='g(n)')
        st.plotly_chart(fig)
         
        fig = go.Figure(data=[go.Bar(x=n_list, y=g)])
        fig.update_layout(title='g(n) Plot', xaxis_title='n', yaxis_title='g(n)')
        st.plotly_chart(fig)

        fig = go.Figure(data=go.Scatter(x=i_list, y=Hw[:len(i_list)]))
        fig.update_layout(title='Hw Plot', xaxis_title='i', yaxis_title='Gw')
        st.plotly_chart(fig)
       
        fig = go.Figure(data=go.Scatter(x=i_list, y=Gw[:len(i_list)]))
        fig.update_layout(title='Gw Plot', xaxis_title='i', yaxis_title='Gw')
        st.plotly_chart(fig)
     

        for i in range(1, 9):
            trace = go.Scatter(x=i_list, y=Q[i], mode='lines', name=f'Q[{i}]')
            traces.append(trace)
            
            
            layout = go.Layout(title='Qj (f)',
                               xaxis=dict(title=''),
                               yaxis=dict(title=''))
            
            
            fig = go.Figure(data=traces, layout=layout)
            st.plotly_chart(fig)
   

            qj = np.zeros((6, 10000))
            k_list = []
            j = 1
            
            # Calculations
            a = -(round (2*j) + round (2*(j-1)) - 2 )
            st.write(f"a = {a}")
            b=-(1- round(2**(j-1)))+1
            st.write(f"b  = {b}")
           
            
        for k in range(a, b):
            k_list.append(k)
            qj[1][k + abs(a)] = -2 * (dirac(k) - dirac(k+1))
            # Visualization using Plotly
            fig = go.Figure(data=[go.Bar(x=k_list, y=qj[1][0:len(k_list)])])
            fig.update_layout(title='q1(k)', xaxis_title='', yaxis_title='')
            
            st.plotly_chart(fig)
 
            k_list2 = []
            j2 = 2
            a2 = -(round(2*j2) + round(2*(j2-1)) - 2)
            st.write(f"a = {a2}")
            b2 = -(1 - round(2**(j2-1))) + 1
            st.write(f"b  = {b2}")
            
        for k in range(a2, b2):
            k_list2.append(k)
            qj[2][k + abs(a2)] = -1/4 * (dirac(k-1) + 3*dirac(k) + 2*dirac(k+1) - 2*dirac(k+2) - 3*dirac(k+3) - dirac(k+4))
        
            fig2 = go.Figure(data=[go.Bar(x=k_list2, y=qj[2][0:len(k_list2)])])
            fig2.update_layout(title='q2(k)', xaxis_title='', yaxis_title='')
            st.plotly_chart(fig2)
         
            k_list3 = []
            j3 = 3
            a3 = -(round(2*j3) + round(2*(j3-1)) - 2)
            st.write(f"a = {a3}")
            b3 = -(1 - round(2**(j3-1))) + 1
            st.write(f"b  = {b3}")
                
        for k in range(a3, b3):
            k_list3.append(k)
            qj[3][k + abs(a3)] = -1/32 * (dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k)
                                                  + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4) - 9*dirac(k+5)
                                                  - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8) - 3*dirac(k+9) - dirac(k+10))
            
            fig3 = go.Figure(data=[go.Bar(x=k_list3, y=qj[3][0:len(k_list3)])])
            fig3.update_layout(title='q3(k)', xaxis_title='', yaxis_title='')
            st.plotly_chart(fig3)

            k_list4 = []
            j4 = 4
            a4 = -(round(2*j4) + round(2*(j4-1)) - 2)
            st.write(f"a  = {a4}")
            b4 = -(1 - round(2**(j4-1))) + 1
            st.write(f"b  = {b4}")
                
        for k in range(a4, b4):
            k_list4.append(k)
            qj[4][k + abs(a4)] = -1/256 * (dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4) + 15*dirac(k-3)
                                                   + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k) + 41*dirac(k+1) + 43*dirac(k+2)
                                                   + 42*dirac(k+3) + 38*dirac(k+4) + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7)
                                                   - 8*dirac(k+8) - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
                                                   - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16) - 21*dirac(k+17)
                                                   - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20) - 3*dirac(k+21) - dirac(k+22))
            
            fig4 = go.Figure(data=[go.Bar(x=k_list4, y=qj[4][0:len(k_list4)])])
            fig4.update_layout(title='q4(k)', xaxis_title='', yaxis_title='')
            st.plotly_chart(fig4)

                
            k_list5 = []
            j5 = 5
            a5 = -(round(2*j5) + round(2*(j5-1)) - 2)
            st.write(f"a = {a5}")
            b5 = -(1 - round(2**(j5-1))) + 1
            st.write(f"b  = {b5}")
            
        for k in range(a5, b5):
            k_list5.append(k)
            qj[5][k + abs(a5)] = -1/512 * (dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
                                               + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
                                               + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
                                               + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
                                               + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45*dirac(k+14)
                                               + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20)
                                               - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
                                               - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
                                               - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
                                               - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
                                               - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
                                               - dirac(k+46))
        
            fig5 = go.Figure(data=[go.Bar(x=k_list5, y=qj[5][0:len(k_list5)])])
            fig5.update_layout(title='Fifth Part', xaxis_title='', yaxis_title='')
            st.plotly_chart(fig5)
