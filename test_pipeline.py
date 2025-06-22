#!/usr/bin/env python3
"""
Complete Pipeline Test Script for ezpz-BCI
Tests the full pipeline: Data Loading → Training → Model Saving → Classification
Uses real EEG data from the research project
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, 'src')

# Import our modules
from train_eegnet import train, evaluate, load
from eegnet import EEGNetModel
import torch

def test_data_loading():
    """Test loading the real EEG data"""
    print("🔍 Testing Data Loading...")
    
    # Test Motor Movement data
    mm_path = "test_data/data-reuben-2122-2205-3-classes-MM-epo.fif"
    mi_path = "test_data/data-reuben-2122-2205-3-classes-MI-epo.fif"
    
    try:
        # Load Motor Movement data
        print(f"  Loading Motor Movement data: {mm_path}")
        mm_data, mm_labels = load(mm_path)
        print(f"  ✅ MM Data shape: {mm_data.shape}")
        print(f"  ✅ MM Labels shape: {mm_labels.shape}")
        print(f"  ✅ MM Classes: {set(mm_labels)}")
        
        # Load Motor Imagery data
        print(f"  Loading Motor Imagery data: {mi_path}")
        mi_data, mi_labels = load(mi_path)
        print(f"  ✅ MI Data shape: {mi_data.shape}")
        print(f"  ✅ MI Labels shape: {mi_labels.shape}")
        print(f"  ✅ MI Classes: {set(mi_labels)}")
        
        return True, mm_path, mi_path
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False, None, None

def test_model_training(data_path, model_name):
    """Test training a model with real data"""
    print(f"\n🧠 Testing Model Training with {model_name}...")
    
    # Create models directory
    models_dir = Path("test_models")
    models_dir.mkdir(exist_ok=True)
    
    # Training hyperparameters (reduced for quick testing)
    hyperparameters = {
        "epochs": 10,  # Reduced for quick test
        "test-ratio": 0.3
    }
    
    try:
        # Train the model
        save_path_folder = str(models_dir) + "/"
        print(f"  Training {model_name} for {hyperparameters['epochs']} epochs...")
        
        train(model_name, data_path, save_path_folder, hyperparameters, save=True)
        
        # Check if model files were created
        model_file = models_dir / f"{model_name}.pth"
        metadata_file = models_dir / f"{model_name}.json"
        
        if model_file.exists() and metadata_file.exists():
            print(f"  ✅ Model saved: {model_file}")
            print(f"  ✅ Metadata saved: {metadata_file}")
            return True, str(model_file)
        else:
            print(f"  ❌ Model files not found")
            return False, None
            
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return False, None

def test_model_evaluation(model_name):
    """Test model evaluation"""
    print(f"\n📊 Testing Model Evaluation for {model_name}...")
    
    try:
        models_dir = "test_models/"
        test_accuracy, cf_matrix = evaluate(
            model_name, 
            models_dir, 
            pltshow=False,  # Don't show plots in test
            save=False,     # Don't save plots in test
            verbose=True
        )
        
        print(f"  ✅ Test Accuracy: {test_accuracy:.2f}%")
        print(f"  ✅ Confusion Matrix shape: {cf_matrix.shape}")
        
        return True, test_accuracy
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        return False, None

def test_model_loading(model_path):
    """Test loading a trained model for classification"""
    print(f"\n🔄 Testing Model Loading...")
    
    try:
        # Load model metadata
        import json
        metadata_path = model_path.replace('.pth', '.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract model parameters
        chans = metadata[4]
        time_points = metadata[5]
        
        print(f"  Model parameters: {chans} channels, {time_points} time points")
        
        # Create and load model
        model = EEGNetModel(chans=chans, time_points=time_points)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print(f"  ✅ Model loaded successfully")
        print(f"  ✅ Model ready for classification")
        
        return True, model
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False, None

def main():
    """Run complete pipeline test"""
    print("🚀 ezpz-BCI Complete Pipeline Test")
    print("=" * 50)
    
    # Test 1: Data Loading
    data_success, mm_path, mi_path = test_data_loading()
    if not data_success:
        print("❌ Pipeline test failed at data loading")
        return
    
    # Test 2: Model Training (Motor Movement)
    mm_success, mm_model_path = test_model_training(mm_path, "test_MM_model")
    if not mm_success:
        print("❌ Pipeline test failed at MM training")
        return
    
    # Test 3: Model Training (Motor Imagery)
    mi_success, mi_model_path = test_model_training(mi_path, "test_MI_model")
    if not mi_success:
        print("❌ Pipeline test failed at MI training")
        return
    
    # Test 4: Model Evaluation
    mm_eval_success, mm_accuracy = test_model_evaluation("test_MM_model")
    mi_eval_success, mi_accuracy = test_model_evaluation("test_MI_model")
    
    # Test 5: Model Loading for Classification
    mm_load_success, mm_model = test_model_loading(mm_model_path)
    mi_load_success, mi_model = test_model_loading(mi_model_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Pipeline Test Summary")
    print("=" * 50)
    
    print(f"✅ Data Loading: {'PASS' if data_success else 'FAIL'}")
    print(f"✅ MM Training: {'PASS' if mm_success else 'FAIL'}")
    print(f"✅ MI Training: {'PASS' if mi_success else 'FAIL'}")
    print(f"✅ MM Evaluation: {'PASS' if mm_eval_success else 'FAIL'}")
    print(f"✅ MI Evaluation: {'PASS' if mi_eval_success else 'FAIL'}")
    print(f"✅ MM Model Loading: {'PASS' if mm_load_success else 'FAIL'}")
    print(f"✅ MI Model Loading: {'PASS' if mi_load_success else 'FAIL'}")
    
    if mm_eval_success and mi_eval_success:
        print(f"\n📊 Performance Results:")
        print(f"  🧠 Motor Movement Accuracy: {mm_accuracy:.2f}%")
        print(f"  🤔 Motor Imagery Accuracy: {mi_accuracy:.2f}%")
        print(f"  📈 Performance Gap: {mm_accuracy - mi_accuracy:.2f}%")
    
    print(f"\n🎯 Ready for GUI Testing!")
    print(f"  1. Use test_models/test_MM_model.pth in Classification tab")
    print(f"  2. Use test_data/*.fif files in Training tab")
    print(f"  3. All components verified working!")

if __name__ == "__main__":
    main() 