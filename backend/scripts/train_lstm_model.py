"""
Training Script for LSTM Exercise Classifier
Optimized for RTX 3060 with Mixed Precision Training
Expected: 90-95% accuracy, ~2.5GB VRAM usage
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_exercise_classifier import LSTMExerciseClassifier
import tensorflow as tf


def find_training_videos(data_dir: str):
    """
    Find all training videos organized by exercise type.
    
    Expected directory structure:
    data_dir/
        push_up/
            video1.mp4
            video2.mp4
        squat/
            video1.mp4
        ...
    
    Args:
        data_dir: Root directory containing exercise folders
        
    Returns:
        Tuple of (video_paths, labels)
    """
    video_paths = []
    labels = []
    
    exercise_types = [
        'push_up',
        'squat',
        'barbell_biceps_curl',
        'hammer_curl',
        'shoulder_press'
    ]
    
    print(f"🔍 Searching for training videos in: {data_dir}")
    print("="*60)
    
    for exercise in exercise_types:
        exercise_dir = os.path.join(data_dir, exercise)
        
        if not os.path.exists(exercise_dir):
            print(f"⚠️  Directory not found: {exercise_dir}")
            continue
        
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(exercise_dir, ext)))
        
        if len(videos) == 0:
            print(f"⚠️  No videos found for: {exercise}")
            continue
        
        print(f"✅ {exercise}: {len(videos)} videos")
        
        video_paths.extend(videos)
        labels.extend([exercise] * len(videos))
    
    print("="*60)
    print(f"📊 Total videos found: {len(video_paths)}")
    
    return video_paths, labels


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM Exercise Classifier on RTX 3060'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing training videos organized by exercise type'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=50,
        help='Number of frames per sequence (default: 50)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32, use 64 for faster training)'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation (default: 0.2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained model (default: models)'
    )
    parser.add_argument(
        '--no-mixed-precision',
        action='store_true',
        help='Disable mixed precision training (slower but more stable)'
    )
    parser.add_argument(
        '--export-tflite',
        action='store_true',
        help='Export model to TensorFlow Lite after training'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        print("\n📝 Expected directory structure:")
        print("   data_dir/")
        print("       push_up/")
        print("           video1.mp4")
        print("           video2.mp4")
        print("       squat/")
        print("           video1.mp4")
        print("       ...")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*60)
    print("🏋️  LSTM EXERCISE CLASSIFIER TRAINING")
    print("="*60)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Sequence length: {args.sequence_length} frames")
    print(f"🔄 Max epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"✂️  Validation split: {args.validation_split * 100}%")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"⚡ Mixed precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}")
    print("="*60 + "\n")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️  No GPU detected. Training will be slower on CPU.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            sys.exit(0)
    
    print()
    
    # Find training videos
    video_paths, labels = find_training_videos(args.data_dir)
    
    if len(video_paths) == 0:
        print("❌ No training videos found. Please check your data directory.")
        sys.exit(1)
    
    # Minimum videos per class check
    from collections import Counter
    label_counts = Counter(labels)
    min_samples = min(label_counts.values())
    
    if min_samples < 10:
        print(f"\n⚠️  Warning: Some classes have fewer than 10 videos.")
        print(f"   Minimum samples: {min_samples}")
        print("   Recommendation: At least 20-30 videos per exercise for good accuracy.")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            sys.exit(0)
    
    print()
    
    # Initialize classifier
    classifier = LSTMExerciseClassifier(
        sequence_length=args.sequence_length,
        num_classes=5,
        use_mixed_precision=not args.no_mixed_precision
    )
    
    # Build model
    print("🏗️  Building LSTM model...")
    classifier.build_model()
    classifier.get_model_summary()
    
    print()
    
    # Prepare training data
    print("📚 Preparing training data...")
    print("   This may take several minutes depending on video count and length...")
    print()
    
    X_train, y_train, X_val, y_val = classifier.prepare_training_data(
        video_paths=video_paths,
        labels=labels,
        validation_split=args.validation_split
    )
    
    print()
    
    # Train model
    model_save_path = os.path.join(args.output_dir, 'lstm_exercise_model.h5')
    
    history = classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=model_save_path
    )
    
    # Save final model and encoder
    encoder_path = os.path.join(args.output_dir, 'lstm_exercise_encoder.pkl')
    classifier.save_model(model_save_path, encoder_path)
    
    # Export to TFLite if requested
    if args.export_tflite:
        print("\n📦 Exporting to TensorFlow Lite...")
        tflite_path = os.path.join(args.output_dir, 'lstm_exercise_model.tflite')
        classifier.export_tflite(
            model_path=model_save_path,
            tflite_path=tflite_path,
            use_float16=True
        )
    
    # Print final summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Model saved to: {model_save_path}")
    print(f"📁 Encoder saved to: {encoder_path}")
    if args.export_tflite:
        print(f"📁 TFLite model saved to: {tflite_path}")
    print("\n📊 Training Summary:")
    print(f"   Total epochs: {len(history.history['loss'])}")
    print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print("\n🚀 Next Steps:")
    print("   1. Test the model with: python scripts/test_lstm_model.py")
    print("   2. Integrate with API: Update main.py to use LSTM classifier")
    print("   3. Deploy to production!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
