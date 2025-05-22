import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from tkinter import Tk, filedialog, simpledialog, messagebox, Listbox, MULTIPLE, END
import os
import re
import time as systime
import json

# --------------------------------
# Helper Functions
# --------------------------------

def find_header_row(filepath):
    for i in range(30):
        row = pd.read_csv(filepath, skiprows=i, nrows=1, header=None)
        row_str = row.iloc[0].astype(str)
        if row_str.str.contains("X\\[s\\]").sum() >= 3 and row_str.str.contains("EMG").sum() >= 1:
            return i
    raise ValueError("Could not find a valid EMG header row.")

def smooth(signal, window_size=200):
    return np.convolve(np.abs(signal), np.ones(window_size)/window_size, mode='same')

def select_file():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Select EMG CSV File")

# Global variable for channel selection
selected_channels = []

def select_channels(channels):
    sel_window = Tk()
    sel_window.title("Select EMG Channels")

    lb = Listbox(sel_window, selectmode=MULTIPLE, width=60)
    lb.pack()
    for ch in channels:
        lb.insert(END, ch)

    def confirm_selection():
        global selected_channels
        selected_channels = [channels[i] for i in lb.curselection()]
        sel_window.quit()

    from tkinter import Button
    Button(sel_window, text="Confirm", command=confirm_selection).pack()
    sel_window.mainloop()
    sel_window.destroy()

# --------------------------------
# Main Script
# --------------------------------

file_path = select_file()
if not file_path:
    raise Exception("No file selected.")

header_row = find_header_row(file_path)
df_headers = pd.read_csv(file_path, skiprows=header_row, nrows=1, header=None)
all_columns = df_headers.iloc[0].tolist()
emg_channels = [col for col in all_columns if "EMG" in str(col)]

select_channels(emg_channels)
if not selected_channels:
    raise Exception("No EMG channels selected.")

root = Tk(); root.withdraw()
expected_reps = simpledialog.askinteger("Input", "Enter expected number of repetitions:", minvalue=1, maxvalue=100)

df = pd.read_csv(file_path, skiprows=header_row, header=0)
fs = 1925.926
time_col_candidates = [c for c in df.columns if "X[s]" in str(c)]
time_col = time_col_candidates[4] if len(time_col_candidates) >= 5 else time_col_candidates[0]

results_all = []
metadata_log = {}

for ch in selected_channels:
    print(f"\n🔍 Analyzing: {ch}")
    time = pd.to_numeric(df[time_col], errors='coerce')
    emg = pd.to_numeric(df[ch], errors='coerce')
    valid = time.notna() & emg.notna()
    time = time[valid].reset_index(drop=True)
    emg = emg[valid].reset_index(drop=True)

    # ✅ Normalize EMG (per channel max)
    max_val = np.max(np.abs(emg))
    emg = emg / max_val if max_val != 0 else emg

    envelope = smooth(emg)

    # ✅ Robust threshold: Median + MAD
    median_env = np.median(envelope)
    mad_env = np.median(np.abs(envelope - median_env))
    threshold = median_env + 6 * mad_env

    metadata_log[ch] = {
        "Threshold Method": "Median + 6 × MAD",
        "Median Envelope": float(median_env),
        "MAD Envelope": float(mad_env),
        "Computed Threshold": float(threshold),
        "Max EMG Value (pre-normalization)": float(max_val),
        "Sampling Frequency (Hz)": fs,
        "Min Rep Duration (s)": 0.3,
        "Expected Repetitions": expected_reps
    }

    # Threshold crossings
    above = envelope > threshold
    segments = []
    min_samples = int(0.3 * fs)
    start = None
    for i, val in enumerate(above):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            if end - start >= min_samples:
                segments.append((start, end))
            start = None

    if len(segments) > expected_reps:
        segments.sort(key=lambda s: max(envelope[s[0]:s[1]]), reverse=True)
        segments = sorted(segments[:expected_reps], key=lambda s: s[0])

    # Feature extraction with rep number
    for idx, (s, e) in enumerate(segments, start=1):
        sig = emg[s:e].to_numpy()
        t = time[s:e].to_numpy()
        rms = np.sqrt(np.mean(sig**2))
        N = len(sig)
        yf = fft(sig)
        xf = fftfreq(N, 1/fs)
        psd = np.abs(yf[:N//2])**2
        freqs = xf[:N//2]
        mf = np.sum(freqs * psd) / np.sum(psd)

        results_all.append({
            'Channel': ch,
            'Rep Number': idx,
            'Start Time (s)': t[0],
            'End Time (s)': t[-1],
            'Duration (s)': t[-1] - t[0],
            'RMS (normalized)': rms,
            'Mean Frequency (Hz)': mf
        })

    results_all.append({})  # Blank row

    # Plot and save
    clean_channel = re.sub(r'[^\w\s-]', '', ch).replace(" ", "_")
    plt.figure(figsize=(13, 4))
    plt.plot(time, emg, label=ch, linewidth=0.6)
    for s, e in segments:
        plt.axvspan(time[s], time[e], color='red', alpha=0.3)
    plt.title(f"Detected Repetitions: {ch}")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plot_filename = os.path.splitext(file_path)[0] + f"_{clean_channel}.png"
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
    print(f"📷 Saved plot for {ch} to {plot_filename}")

# Save results
timestamp = systime.strftime("%Y%m%d_%H%M%S")
base = os.path.splitext(file_path)[0]
csv_path = f"{base}_EMG_Results_{timestamp}.csv"
json_path = f"{base}_EMG_Metadata_{timestamp}.json"

results_df = pd.DataFrame(results_all)
results_df.to_csv(csv_path, index=False)

# Save metadata log
with open(json_path, 'w') as f:
    json.dump(metadata_log, f, indent=4)

messagebox.showinfo("Done", f"✅ Analysis complete.\nResults saved to:\n{csv_path}")
print(results_df)
