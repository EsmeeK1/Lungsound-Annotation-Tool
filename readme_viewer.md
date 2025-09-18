# ECG Signal Viewer

**A Tkinter-based ECG signal viewer designed to display, filter, and annotate ECG signals from multiple leads.**

---
## 📚 Introduction

The **ECG (Electrocardiogram) Signal Viewer** is a tool for visualizing and analyzing ECG signals, primarily designed for biomedical researchers, healthcare professionals, and anyone involved in ECG data analysis. The tool provides a streamlined way to load, view, filter, and annotate ECG signals, helping users examine heart activity with precision and flexibility. 

### What is an ECG Signal?

An ECG (Electrocardiogram) is a recording of the electrical activity of the heart over time. This signal is obtained through electrodes placed on the skin, which detect and measure the electrical impulses that cause the heart to contract and pump blood. These impulses are recorded as waveforms, with each waveform representing a specific part of the cardiac cycle.

### Components of an ECG Signal

The standard ECG waveform is composed of several key features:

- **P Wave**: Represents atrial depolarization, occurring just before the atria contract.
- **QRS Complex**: Represents ventricular depolarization, occurring just before the ventricles contract. This is the most prominent part of the ECG and is used to detect many types of heart abnormalities.
- **T Wave**: Represents ventricular repolarization, the process of the heart muscle preparing for the next contraction.
- **U Wave**: Sometimes seen after the T wave; it is less understood but may represent further repolarization.

### Types of ECG Leads

A typical ECG consists of **12 leads**. Each lead records the heart’s electrical activity from a different angle, providing a comprehensive view of cardiac health. These leads are grouped as follows:

1. **Limb Leads**: Leads I, II, and III measure electrical activity along the vertical plane of the heart.
2. **Augmented Leads**: aVR, aVL, and aVF are recorded from a single electrode on the limbs, with signals enhanced to provide additional perspectives.
3. **Chest Leads**: Leads V1 to V6 are positioned across the chest to provide a view of electrical activity from the front of the heart.

### Applications of ECG Signals

ECG signals are widely used in medicine to assess heart health. Key applications include:

- **Heart Rate Monitoring**: ECG helps measure heart rate and identify tachycardia (fast heart rate) or bradycardia (slow heart rate).
- **Arrhythmia Detection**: Analyzing ECG signals helps detect arrhythmias, such as atrial fibrillation, which can lead to serious complications if untreated.
- **Myocardial Infarction Diagnosis**: ECG is crucial in identifying ischemic events (heart attacks) by detecting ST-segment elevation or depression.
- **Cardiac Condition Assessment**: Conditions like hypertrophy, conduction blocks, and electrolyte imbalances are often visible in ECG patterns.

### Why Use the ECG Signal Viewer?

This application provides a comprehensive platform for exploring 12-lead ECG data. Users can not only visualize ECG signals but also apply filtering, annotate specific events, and save these annotations for further analysis. This flexibility makes it an ideal tool for:

- **Research and Analysis**: Researchers can label data manually, which is essential for machine learning model training and validation.
- **Clinical Review**: Healthcare professionals can use the viewer to explore patterns and trends in patient ECG data.
- **Educational Use**: Medical students and educators can leverage the tool to learn and teach ECG interpretation skills.

By offering these features, the ECG Signal Viewer bridges the gap between data visualization and data labeling, providing a robust tool for ECG analysis and annotation.

---



---

## 🌟 Features

- **Multi-Lead ECG Display**: Visualizes ECG signals from 12 leads, with labeled axes and gridlines for enhanced readability.
- **Filtering Options**: Includes low-pass and notch filters to remove noise and improve signal quality.
- **Interactive Annotation**: Allows users to label rhythms and morphologies for each ECG signal.
- **Data Saving**: Save annotated rhythms and morphologies with timestamps to a CSV file.
- **Signal Navigation**: Easily browse through ECG files in a folder.
- **Custom File Extension Support**: Option to specify file extensions (e.g., `.D20`, `.bin`) before loading files.

---

## 🛠 Requirements

- **Python** 3.7 or higher
- **Required Python Packages**:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `pywavelets`
  - `tkinter` (included with standard Python installations)

Install dependencies with:


`pip install numpy matplotlib scipy pywavelets`


## Usage
Open a Folder: Select the folder containing ECG signal files (e.g., .D20, .D21, .D22, .bin, .ecg).

Select Signal: Choose the signal file you want to view from the dropdown.

Select Rhythm and Morphology: Use the dropdown menus to select the rhythm (e.g., Sinus Rhythm, Atrial Fibrillation) and morphology (e.g., Normal, LBBB).

View ECG Signals: The application will display the ECG signals, with each lead shown in a separate plot. The signals are filtered to remove high-frequency noise and power-line interference.

Navigate Through Signals: Use the "Next" and "Previous" buttons to cycle through the ECG signals in the selected folder.

Save Results: After selecting rhythm and morphology for the signal, click the "Save" button to record the selections in a results.csv file.


## Code Description

## Key Components:

## Signal Processing:

Lowpass Filter: Applied to each lead to remove high-frequency noise.

Power Line Filter: A notch filter applied to remove 60Hz power line interference.

## Tkinter GUI:

Allows the user to open a folder, select signals, and choose rhythm/morphology via dropdown menus.

Displays the ECG signals using Matplotlib, with a vertical scrollbar for navigation.

CSV Export: Saves the selected rhythm and morphology along with a timestamp and filename to a results.csv file.

## Main Functions:

apply_lowpass_filter(signal_data, sample_rate): Applies a low-pass filter to the ECG data.

apply_power_line_filter(signal_data, sample_rate, power_line_frequency): Applies a notch filter to remove power line noise.

show_current_signal(): Displays the current ECG signal on the GUI.

save_results(): Saves the selected rhythm, morphology, and timestamp to a CSV file.

## Troubleshooting

* No signals displayed: Ensure that the selected folder contains ECG files and that the correct signal extension is chosen.

* No response when selecting a signal: Check that the file format is compatible and contains valid ECG data.

#Future Improvements

* Support for additional signal processing techniques (e.g., wavelet transforms).
  
* Enhanced error handling for unsupported file formats.
  
*Ability to display more detailed information about the selected signal.

