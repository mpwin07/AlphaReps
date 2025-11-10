"""
Test Script for LSTM Exercise Classifier
Evaluate model performance and inference speed
"""

import os
import sys
import argparse
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_exercise_classifier import LSTMExerciseClassifier
import tensorflow as tf


def test_inference_speed(classifier, num_iterations=100):
    """
    Test inference speed on GPU and CPU.
    
    Args:
        classifier: Trained LSTM classifier
        num_iterations: Number of test iterations
    """
    print("\n" + "="*60)
    print("⚡ INFERENCE SPEED TEST")
    print("="*60)
    
    # Create dummy sequence
    dummy_sequence = np.random.randn(1, classifier.sequence_length, classifier.feature_dim).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        _ = classifier.model.predict(dummy_sequence, verbose=0)
    
    # GPU/CPU test
    gpus = tf.config.list_physical_devices('GPU')
    device_name = "GPU" if gpus else "CPU"
    
    print(f"\n🔧 Testing on {device_name}...")
    print(f"   Iterations: {num_iterations}")
    
    times = []
    for i in range(num_iterations):
        start = time.time()
        _ = classifier.model.predict(dummy_sequence, verbose=0)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n📊 Results:")
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Min: {min_time:.2f} ms")
    print(f"   Max: {max_time:.2f} ms")
    print(f"   FPS: {1000/avg_time:.1f}")
    
    # Performance evaluation
    if device_name == "GPU":
        if avg_time < 25:
            print(f"\n✅ Excellent! Inference time is within target (< 25ms)")
        elif avg_time < 50:
            print(f"\n⚠️  Good, but slightly slower than target (25ms)")
        else:
            print(f"\n⚠️  Slower than expected. Consider:")
            print(f"      - Using TFLite model")
            print(f"      - Reducing sequence length")
            print(f"      - Checking GPU utilization")
    else:
        if avg_time < 80:
            print(f"\n✅ Good CPU performance (< 80ms)")
        else:
            print(f"\n⚠️  CPU inference is slow. GPU recommended for real-time use.")
    
    print("="*60)


def test_video_prediction(classifier, video_path):
    """
    Test prediction on a single video file.
    
    Args:
        classifier: Trained LSTM classifier
        video_path: Path to test video
    """
    print("\n" + "="*60)
    print("🎥 VIDEO PREDICTION TEST")
    print("="*60)
    print(f"📁 Video: {video_path}")
    
    start_time = time.time()
    predicted_class, confidence = classifier.predict_from_video(video_path)
    end_time = time.time()
    
    print(f"\n📊 Prediction Results:")
    print(f"   Exercise: {predicted_class.upper().replace('_', ' ')}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Processing time: {(end_time - start_time)*1000:.2f} ms")
    print("="*60)


def test_model_accuracy(classifier, test_videos, test_labels):
    """
    Test model accuracy on a test set.
    
    Args:
        classifier: Trained LSTM classifier
        test_videos: List of test video paths
        test_labels: List of ground truth labels
    """
    print("\n" + "="*60)
    print("📊 ACCURACY TEST")
    print("="*60)
    print(f"Testing on {len(test_videos)} videos...")
    
    correct = 0
    total = 0
    predictions = []
    confidences = []
    
    for i, (video_path, true_label) in enumerate(zip(test_videos, test_labels)):
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(test_videos)} videos...")
        
        predicted_class, confidence = classifier.predict_from_video(video_path)
        predictions.append(predicted_class)
        confidences.append(confidence)
        
        if predicted_class == true_label:
            correct += 1
        total += 1
    
    accuracy = (correct / total) * 100
    avg_confidence = np.mean(confidences)
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"   Average Confidence: {avg_confidence*100:.2f}%")
    
    # Per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, true, conf in zip(predictions, test_labels, confidences):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    
    print(f"\n📊 Per-Class Accuracy:")
    for exercise in sorted(class_total.keys()):
        class_acc = (class_correct[exercise] / class_total[exercise]) * 100
        print(f"   {exercise.upper().replace('_', ' ')}: {class_acc:.2f}% ({class_correct[exercise]}/{class_total[exercise]})")
    
    # Performance evaluation
    if accuracy >= 90:
        print(f"\n✅ Excellent! Model achieves target accuracy (≥90%)")
    elif accuracy >= 80:
        print(f"\n⚠️  Good accuracy, but below target (90%)")
        print(f"   Consider:")
        print(f"      - More training data")
        print(f"      - Data augmentation")
        print(f"      - Longer training")
    else:
        print(f"\n❌ Accuracy below expectations")
        print(f"   Recommendations:")
        print(f"      - Check data quality")
        print(f"      - Increase training data")
        print(f"      - Review model architecture")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Test LSTM Exercise Classifier'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/lstm_exercise_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--encoder-path',
        type=str,
        default='models/lstm_exercise_encoder.pkl',
        help='Path to label encoder'
    )
    parser.add_argument(
        '--test-video',
        type=str,
        help='Path to a single video for testing'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        help='Directory containing test videos (same structure as training data)'
    )
    parser.add_argument(
        '--speed-test',
        action='store_true',
        help='Run inference speed test'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations for speed test (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("   Please train the model first using train_lstm_model.py")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🧪 LSTM EXERCISE CLASSIFIER TEST")
    print("="*60)
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Encoder: {args.encoder_path}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {gpus[0].name}")
    else:
        print("⚠️  No GPU detected. Using CPU.")
    
    print("="*60)
    
    # Load model
    print("\n📚 Loading model...")
    classifier = LSTMExerciseClassifier()
    classifier.load_model(args.model_path, args.encoder_path)
    print("✅ Model loaded successfully!")
    
    # Run tests
    if args.speed_test:
        test_inference_speed(classifier, args.iterations)
    
    if args.test_video:
        if not os.path.exists(args.test_video):
            print(f"\n❌ Error: Video not found at {args.test_video}")
        else:
            test_video_prediction(classifier, args.test_video)
    
    if args.test_dir:
        if not os.path.exists(args.test_dir):
            print(f"\n❌ Error: Test directory not found at {args.test_dir}")
        else:
            # Import the function from training script
            from train_lstm_model import find_training_videos
            test_videos, test_labels = find_training_videos(args.test_dir)
            
            if len(test_videos) == 0:
                print("\n❌ No test videos found")
            else:
                test_model_accuracy(classifier, test_videos, test_labels)
    
    # If no tests specified, show usage
    if not (args.speed_test or args.test_video or args.test_dir):
        print("\n📝 No tests specified. Available options:")
        print("   --speed-test          : Test inference speed")
        print("   --test-video PATH     : Test on a single video")
        print("   --test-dir PATH       : Test on a directory of videos")
        print("\nExample:")
        print("   python scripts/test_lstm_model.py --speed-test --test-video data/push_up/video1.mp4")
    
    print()


if __name__ == "__main__":
    main()
