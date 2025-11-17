"""
Test script to verify model integration
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.video_exercise_classifier import VideoExerciseClassifier

def test_model_loading():
    """Test if the trained model can be loaded successfully"""
    print("="*60)
    print("TESTING MODEL INTEGRATION")
    print("="*60)
    
    try:
        # Test auto-loading
        print("\n1. Testing auto-loading...")
        classifier = VideoExerciseClassifier(auto_load_model=True)
        
        if classifier.model_loaded:
            print("SUCCESS: Model auto-loaded successfully!")
            
            # Test model properties
            print(f"Exercise classes: {classifier.exercise_classes}")
            print(f"Label encoder classes: {list(classifier.label_encoder.classes_)}")
            
            # Test if we can make a dummy prediction (with random data)
            print("\n2. Testing prediction capability...")
            import numpy as np
            
            # Create dummy landmarks (146 features)
            dummy_landmarks = np.random.rand(146)
            
            try:
                prediction = classifier.predict(dummy_landmarks)
                print(f"SUCCESS: Prediction test successful: {prediction}")
                
                # Test probability prediction
                probabilities = classifier.predict_proba(dummy_landmarks)
                print(f"SUCCESS: Probability prediction test successful")
                for exercise, prob in probabilities.items():
                    print(f"   {exercise}: {prob:.3f}")
                
                # Test individual model predictions
                individual_preds = classifier.get_individual_predictions(dummy_landmarks)
                print(f"Individual model predictions:")
                for model_name, pred_info in individual_preds.items():
                    print(f"   {model_name}: {pred_info['prediction']} (confidence: {pred_info['confidence']:.3f})")
                
            except Exception as e:
                print(f"ERROR: Prediction test failed: {e}")
                return False
                
        else:
            print("ERROR: Model auto-loading failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nSUCCESS: All tests passed! Model integration is working correctly.")
    return True

def test_unified_system():
    """Test the unified workout system"""
    print("\n" + "="*60)
    print("TESTING UNIFIED WORKOUT SYSTEM")
    print("="*60)
    
    try:
        from unified_workout_system import UnifiedWorkoutSystem
        
        print("\n3. Testing UnifiedWorkoutSystem initialization...")
        system = UnifiedWorkoutSystem()
        
        print("SUCCESS: UnifiedWorkoutSystem initialized successfully!")
        print(f"Available counters: {list(system.counters.keys())}")
        print(f"Current exercise: {system.current_exercise}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: UnifiedWorkoutSystem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting model integration tests...\n")
    
    # Test 1: Model loading
    model_test = test_model_loading()
    
    # Test 2: Unified system
    if model_test:
        system_test = test_unified_system()
        
        if system_test:
            print("\nALL TESTS PASSED!")
            print("SUCCESS: Model integration is complete and working!")
            print("\nReady to start workout sessions!")
            print("   Run: python start_workout.py")
        else:
            print("\nERROR: System integration test failed")
    else:
        print("\nERROR: Model loading test failed")
