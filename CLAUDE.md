# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EZPZ-BCI is a brain-computer interface application that translates EEG signals into computer inputs using machine learning. Users train the system to recognize distinct brainwave patterns (motor imagery tasks) and map them to configurable keyboard commands.

## Development Commands

### Environment Setup
```bash
# Create conda environment from specification
conda env create -f environment.yml

# Activate environment
conda activate ezpz-bci
```

### Running the Application
```bash
# Start the GUI application
python start.py
```

### Core Dependencies
- **PySide6**: GUI framework
- **PyTorch**: Neural network implementation
- **MNE (v1.9.0)**: EEG signal processing
- **BrainFlow**: Hardware interfacing
- **pynput**: Keyboard input simulation

## Architecture Overview

### Entry Point
- `start.py`: Main launcher that configures Python path and starts GUI

### Core Components

#### Backend (`src/backend/`)
- **`eegnet.py`**: EEGNet CNN model implementation with `EEGNetModel`, `TrainModel`, and `EvalModel` classes
- **`classify.py`**: Real-time classification engine that processes live EEG data and triggers keyboard actions
- **`train_eegnet.py`**: Model training pipeline using MNE epochs data
- **`connect.py`**: Device connection utilities for OpenBCI via BrainFlow
- **`key_actuate.py`**: Keyboard simulation using pynput
- **Data processing modules**: Collection and processing for OpenBCI and PhysioNet datasets

#### GUI (`src/gui/`)
- **`home.py`**: Main window with tabbed interface (`MainWindow` and `Home` classes)
- **Core pages**: Device connection, configuration, training, data collection, visualization
- Uses PySide6 with file system watchers for config updates

### Data Flow
1. **Training**: EEG data → MNE preprocessing → EEGNet training → Model persistence
2. **Classification**: Live EEG → Signal processing → Model inference → Key mapping → Keyboard actuation

## Configuration Files

### Core Configuration (`data/`)
- **`config.json`**: Key mappings (e.g., `{"F": "Right Fist"}`)
- **`categories.json`**: Training parameters including epoch_count, learning_rate, categories, device settings
- **`eeg_samples.json`** & **`eeg_markers.json`**: Training data storage
- **`ezpz-test-epo.fif`**: MNE epochs file for testing
- **Model files**: `ezpz-model.pth` & `ezpz-model.json`

## Hardware Support
- OpenBCI Cyton Board
- Neurosity Crown
- Synthetic board (for testing without hardware)

## Key Implementation Details

### EEGNet Model
- Custom PyTorch implementation with temporal/spatial filtering
- Supports motor imagery classification
- Model persistence through PyTorch serialization

### Real-time Processing
- Threading architecture for continuous EEG streaming
- BrainFlow integration for hardware communication
- Real-time classification with configurable confidence thresholds

### GUI Architecture
- Tabbed interface pattern with modular page components
- File system watchers for dynamic configuration updates
- PySide6 Qt integration with custom styling

## Development Notes

### Path Configuration
The `start.py` script automatically adds the project root to Python path, enabling proper imports during development.

### Data Processing Pipeline
MNE is used for EEG signal preprocessing and epoch management. The system expects data in MNE-compatible formats for training.

### Hardware Abstraction
BrainFlow provides a unified interface for different EEG devices, with device selection configured in `categories.json`.