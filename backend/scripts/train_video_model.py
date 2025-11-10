"""
Training Script for Video Exercise Classifier
Trains the AI model to detect different exercises
"""
import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.video_exercise_classifier import VideoExerciseClassifier

async def main():
    print("="*60)
    print("  🏋️  ALPHAREPS MODEL TRAINING")
    print("="*60)
    print()
    
    # Check if dataset exists
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        print(f"   Expected location: {os.path.abspath(dataset_path)}")
        print()
        print("📝 Please create the dataset folder with exercise videos:")
        print("   backend/dataset/")
        print("   ├── barbell_biceps_curl/  (10+ videos)")
        print("   ├── hammer_curl/          (10+ videos)")
        print("   ├── push_up/              (10+ videos)")
        print("   ├── shoulder_press/       (10+ videos)")
        print("   └── squat/                (10+ videos)")
        print()
        return
    
    # Check if dataset has videos
    exercise_folders = ['barbell_biceps_curl', 'hammer_curl', 'push_up', 'shoulder_press', 'squat']
    total_videos = 0
    
    print("📊 Checking dataset...")
    for exercise in exercise_folders:
        exercise_path = os.path.join(dataset_path, exercise)
        if os.path.exists(exercise_path):
            videos = [f for f in os.listdir(exercise_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            total_videos += len(videos)
            print(f"   {exercise}: {len(videos)} videos")
        else:
            print(f"   ⚠️  {exercise}: folder not found")
    
    print()
    
    if total_videos == 0:
        print("❌ No videos found in dataset!")
        print()
        print("📝 Please add exercise videos to:")
        print(f"   {os.path.abspath(dataset_path)}")
        print()
        print("   Each exercise folder should contain 10+ videos")
        print("   Videos should be 5-30 seconds long")
        print()
        return
    
    print(f"✅ Found {total_videos} total videos")
    print()
    
    # Initialize classifier
    print("🔄 Initializing Video Exercise Classifier...")
    classifier = VideoExerciseClassifier()
    
    # Train model
    print()
    print("🎯 Starting training...")
    print("   This may take 10-30 minutes depending on dataset size")
    print()
    
    try:
        accuracy = await classifier.train_model(dataset_path)
        
        print()
        print("="*60)
        print("  ✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"  Final Accuracy: {accuracy:.2%}")
        print()
        print("📁 Model saved to: models/video_exercise_model.pkl")
        print()
        print("🚀 Next steps:")
        print("   1. Test the model:")
        print("      cd backend")
        print("      python start_workout.py")
        print()
        print("   2. Or use directly:")
        print("      python unified_workout_system.py")
        print()
        
    except Exception as e:
        print()
        print("="*60)
        print("  ❌ TRAINING FAILED")
        print("="*60)
        print(f"  Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("💡 Common issues:")
        print("   - Not enough videos (need 10+ per exercise)")
        print("   - Video format not supported (use .mp4)")
        print("   - Corrupted video files")
        print("   - Insufficient memory")
        print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Training cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
