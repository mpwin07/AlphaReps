from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from typing import Dict, Any, Optional
from datetime import datetime
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our models
from models.video_exercise_classifier import VideoExerciseClassifier
from rep_counters.pushup_counter import PushupCounter
from rep_counters.squat_counter import SquatCounter
from rep_counters.curl_counter import CurlCounter
from rep_counters.shoulder_press_counter import ShoulderPressCounter

# Pydantic models
class LoginRequest(BaseModel):
    name: str
    role: str = "user"

class WorkoutFrame(BaseModel):
    frame: str  # base64 encoded image

class User(BaseModel):
    name: str
    role: str
    email: str
    joinDate: str

# FastAPI app
app = FastAPI(
    title="AlphaReps API",
    description="AI-Powered Gym Trainer API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Initialize classifier and counters
classifier = None
counters = {
    'push_up': PushupCounter(),
    'squat': SquatCounter(),
    'barbell_biceps_curl': CurlCounter(),
    'hammer_curl': CurlCounter(),
    'shoulder_press': ShoulderPressCounter()
}

# Session storage (in-memory for demo)
user_sessions = {}
current_exercise = None
current_counter = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global classifier
    print("="*60)
    print("  🚀 ALPHAREPS API STARTING")
    print("="*60)
    
    # Load exercise classifier
    model_path = "models/video_exercise_model.pkl"
    if os.path.exists(model_path):
        print("📚 Loading exercise classifier...")
        classifier = VideoExerciseClassifier()
        classifier.load_model(model_path)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Model not found. Please train the model first.")
        print("   Run: python backend/scripts/train_video_model.py")
        classifier = None
    
    print("✅ AlphaReps API Ready!")
    print("📡 Listening on http://localhost:8000")
    print("="*60)

@app.get("/")
async def root():
    return {
        "message": "Welcome to AlphaReps - AI-Powered Gym Trainer! 💪",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": classifier is not None,
        "features": [
            "Real-time exercise detection",
            "Automatic rep counting",
            "Posture correction",
            "5 supported exercises",
            "95%+ AI accuracy"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }

# Authentication endpoints
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Simple login endpoint"""
    user_data = {
        "name": request.name,
        "role": request.role,
        "email": f"{request.name.lower().replace(' ', '')}@alphareps.com",
        "joinDate": datetime.now().strftime("%b %Y")
    }
    
    # Store session
    user_sessions[request.name] = {
        "user": user_data,
        "login_time": datetime.now(),
        "workouts": []
    }
    
    return {
        "success": True,
        "user": user_data,
        "message": f"Welcome, {request.name}!"
    }

# Workout endpoints
@app.post("/api/workout/analyze")
async def analyze_frame(data: WorkoutFrame):
    """Analyze a single workout frame"""
    global current_exercise, current_counter
    
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.frame.split(',')[1] if ',' in data.frame else data.frame)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get pose landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                "success": False,
                "exercise": "NONE",
                "reps": 0,
                "stage": "ready",
                "feedback": "No person detected. Position yourself in frame.",
                "confidence": 0,
                "formQuality": "UNKNOWN",
                "angles": {}
            }
        
        # Extract landmarks for classification
        landmarks = classifier.extract_landmarks_from_frame(frame)
        
        if landmarks is None:
            return {
                "success": False,
                "exercise": "NONE",
                "reps": 0,
                "stage": "ready",
                "feedback": "Could not extract landmarks",
                "confidence": 0,
                "formQuality": "UNKNOWN",
                "angles": {}
            }
        
        # Classify exercise
        detected_exercise = classifier.predict(landmarks)
        
        # Switch counter if exercise changed
        if detected_exercise != current_exercise:
            current_exercise = detected_exercise
            current_counter = counters.get(detected_exercise)
            if current_counter:
                current_counter.reset()
        
        # Count reps
        if current_counter:
            result = current_counter.count_rep(results.pose_landmarks)
            
            # Parse result based on counter type
            if isinstance(result, tuple):
                if len(result) == 6:  # PushupCounter
                    reps, stage, elbow_angle, feedback, back_angle, form_quality = result
                    angles = {"elbow": elbow_angle, "back": back_angle}
                elif len(result) == 3:  # Standard counters
                    reps, stage, angle = result
                    feedback = f"{stage.upper()}" if stage else "READY"
                    form_quality = "GOOD"
                    angles = {"primary": angle if angle else 0}
                else:
                    reps, stage = 0, "ready"
                    feedback = "Processing..."
                    form_quality = "UNKNOWN"
                    angles = {}
            else:
                reps, stage = 0, "ready"
                feedback = "Processing..."
                form_quality = "UNKNOWN"
                angles = {}
        else:
            reps, stage = 0, "ready"
            feedback = f"Detecting {detected_exercise}..."
            form_quality = "UNKNOWN"
            angles = {}
        
        return {
            "success": True,
            "exercise": detected_exercise.upper().replace('_', ' '),
            "reps": reps,
            "stage": stage,
            "feedback": feedback,
            "confidence": 95,  # Placeholder
            "formQuality": form_quality,
            "angles": angles
        }
        
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# User stats endpoints
@app.get("/api/user/stats")
async def get_user_stats():
    """Get user workout statistics"""
    return {
        "totalWorkouts": 24,
        "totalReps": 1240,
        "avgAccuracy": 94,
        "streak": 7,
        "recentWorkouts": [
            {
                "exercise": "Push-ups",
                "reps": 50,
                "accuracy": 96,
                "date": "Today",
                "time": "10:30 AM"
            },
            {
                "exercise": "Squats",
                "reps": 60,
                "accuracy": 94,
                "date": "Today",
                "time": "09:15 AM"
            }
        ]
    }

@app.get("/api/user/workouts")
async def get_user_workouts():
    """Get user workout history"""
    return {
        "workouts": [
            {
                "id": 1,
                "exercise": "Push-ups",
                "reps": 50,
                "accuracy": 96,
                "date": "2025-10-31",
                "duration": 300
            }
        ]
    }

# Admin endpoints
@app.get("/api/admin/members")
async def get_all_members():
    """Get all gym members"""
    return {
        "members": [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "workouts": 24,
                "accuracy": 95,
                "status": "active",
                "joined": "2 days ago"
            },
            {
                "id": 2,
                "name": "Sarah Smith",
                "email": "sarah@example.com",
                "workouts": 18,
                "accuracy": 92,
                "status": "active",
                "joined": "1 week ago"
            }
        ]
    }

@app.get("/api/admin/stats")
async def get_gym_stats():
    """Get gym-wide statistics"""
    return {
        "totalMembers": 142,
        "activeToday": 48,
        "totalWorkouts": 1248,
        "avgAccuracy": 93
    }

@app.get("/api/exercises")
async def get_supported_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            "push_up",
            "squat",
            "barbell_biceps_curl",
            "hammer_curl",
            "shoulder_press"
        ],
        "total_count": 5
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting AlphaReps API Server...\n")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
