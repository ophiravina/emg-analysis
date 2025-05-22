#  EMG Repetition Analysis Tool
Python tool for EMG repetition detection and feature extraction
tool for automated segmentation and analysis of surface EMG signals collected with Trigno™ sensors.  
Designed for research applications across multiple subjects and sessions.

---

## 📦 Features

- Multi-channel EMG support (e.g., Trigno sensor 5, 6, 7)
- Robust adaptive thresholding (Median + MAD)
- Noise-resistant repetition detection
- RMS and Mean Frequency extraction
- Normalization per channel
- Automatic plotting and CSV export
- Research-ready metadata logging (`.json`)

---

## 🖥️ Installation

### 🔹 Requirements

- Python 3.8 or higher (tested with Python 3.10)

### 🔹 Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/ophiravina/emg-analysis.git
   cd emg-analysis
2. Install required packages:
   pip install pandas matplotlib numpy scipy
## ▶️ Usage
1. Open a terminal in the project directory

2. Run the main script:
   python emg_analysis_gui.py
3. Follow the prompts:
   - Select your .csv EMG file
   - Choose one or more EMG channels
   - Enter expected number of repetitions
## 📁 Outputs
For each run, the script generates:
1. *_EMG_Results_<timestamp>.csv
   → One row per repetition with channel, repetition number, RMS, and mean frequency 
2. *_EMG_Metadata_<timestamp>.json
   → Metadata per channel: threshold method, values, normalization info
3. *_<channel>.png
   → Plot showing raw EMG and detected repetitions
All files are saved in the same folder as your input CSV.

## 📖 Methods
This tool implements signal processing steps aligned with current EMG research best practices:
1. Envelope smoothing using a moving average
2. Robust thresholding (Median + 6 × MAD)
3. Duration-based segment filtering (≥ 0.3s)
4. FFT-based Mean Frequency calculation
5. Root Mean Square (RMS) of normalized EMG
6. Optional repetition count validation
7. Results and plots saved for reproducibility

## 📜 License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
Use for academic and research purposes is permitted with proper attribution.
Commercial use is prohibited without explicit written permission from the author.
🔗 Read the full license
📧 Contact for licensing: ophiravina@gmail.com

## 🧠 Author
Ophir Ravina
📧 ophiravina@gmail.com

## 🙏 Citation
If you use this tool in your academic work, please cite:
Ravina, O. (2025). EMG Repetition Analysis Tool. GitHub repository: https://github.com/ophiravina/emg-analysis

