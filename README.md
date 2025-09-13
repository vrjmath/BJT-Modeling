# Data-Driven Compact Modeling of Bipolar Junction Transistors with Recurrent Neural Networks

[Technical Report] (https://www.osti.gov/servlets/purl/1888718)

## Overview
This project investigates the use of **discrete-time recurrent neural networks (DTRNNs)** as a data-driven alternative to traditional physics-based compact models for circuit simulation.  

The case study focuses on modeling an **Bipolar Junction Transistors (BJT)** and compares the performance of the DTRNN model against the **SPICE Gummel-Poon (SGP)** model.  


## Usage

python main.py 

---

## Output
Running main.py trains the RNN model and computes MSE and NRMSE. Sample waveforms between true and predicted currents will be plotted.
