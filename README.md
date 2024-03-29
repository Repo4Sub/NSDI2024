# NN-defined Modulator
## Overview
The repository contains the majority of the implementation of NN-defined modulators. 
* All three `.ipynb` files in the root directory have the NN-defined modulator template, named as `ModulatorNet()`.
  * The development requires `PyTorch` library and some other commonly-used python modules, like `SciPy`and `NumPy`.
* `QAM_Training` and `OFDM_Training` shows the training procedure of the NN-defined modulators. The training sets are located in `TrainingWaveform/`.
* `TemplateForONXX` demonstrates how to export the `torch` models to `.onnx` models.
* `CodeOnJetson/` contains the files related to the portable deployment of NN-defined modualtors. 
  * `Potability/` contains the `.onnx` models of QAM and OFDM modulators as well as the corresponding input symbols for test.
  *  `WiFi/` and `ZigBee/` folders contain the evaluation files for Over-the-Air transmission using SDR front-end.
