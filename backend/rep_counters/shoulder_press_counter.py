"""
Rep counter for shoulder press
"""
from .base_counter import BaseRepCounter

class ShoulderPressCounter(BaseRepCounter):
    """Counter for shoulder press with improved accuracy"""
    
    def __init__(self):
        super().__init__()
        self.angle_history = []
        self.history_size = 5
        self.min_down_angle = 90   # Must go below 90 degrees (more lenient)
        self.min_up_angle = 140    # Must extend above 140 degrees (more lenient)
        self.debounce_frames = 2   # Reduced for faster response
        self.frames_in_position = 0
        
    def count_rep(self, landmarks):
        """
        Count reps for shoulder press
        Returns: (rep_count, stage, angle)
        """
        try:
            # Get coordinates for right arm
            shoulder = [
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            elbow = [
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]
            wrist = [
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            ]
            
            # Calculate elbow angle
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Smooth angle using moving average
            self.angle_history.append(angle)
            if len(self.angle_history) > self.history_size:
                self.angle_history.pop(0)
            smoothed_angle = sum(self.angle_history) / len(self.angle_history)
            
            # Rep counting logic with debouncing
            if smoothed_angle < self.min_down_angle:
                self.stage = "down"
                self.frames_in_position = 0
                    
            elif smoothed_angle > self.min_up_angle:
                if self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                    self.frames_in_position = 0
            
            return self.counter, self.stage, int(smoothed_angle)
            
        except Exception as e:
            print(f"Error in shoulder press counter: {e}")
            return self.counter, self.stage, None
