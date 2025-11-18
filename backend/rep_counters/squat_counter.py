"""
Rep counter for squats
"""
from .base_counter import BaseRepCounter

class SquatCounter(BaseRepCounter):
    """Counter for squats with improved accuracy"""
    
    def __init__(self):
        super().__init__()
        self.angle_history = []
        self.history_size = 3  # Reduced for faster response
        self.min_down_angle = 90  # More lenient - must go below 90 degrees
        self.min_up_angle = 160   # More lenient - must go above 160 degrees
        self.debounce_frames = 1  # Minimal debouncing for immediate response
        self.frames_in_position = 0
        
    def count_rep(self, landmarks):
        """
        Count reps for squats
        Returns: (rep_count, stage, angle)
        """
        try:
            # Get coordinates for right leg
            hip = [
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            knee = [
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ]
            ankle = [
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]
            
            # Calculate knee angle
            angle = self.calculate_angle(hip, knee, ankle)
            
            # Smooth angle using moving average
            self.angle_history.append(angle)
            if len(self.angle_history) > self.history_size:
                self.angle_history.pop(0)
            smoothed_angle = sum(self.angle_history) / len(self.angle_history)
            
            # Rep counting logic with debouncing
            if smoothed_angle > self.min_up_angle:
                self.stage = "up"
                self.frames_in_position = 0
                    
            elif smoothed_angle < self.min_down_angle:
                if self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                    self.frames_in_position = 0
            
            return self.counter, self.stage, int(smoothed_angle)
            
        except Exception as e:
            print(f"Error in squat counter: {e}")
            return self.counter, self.stage, None
