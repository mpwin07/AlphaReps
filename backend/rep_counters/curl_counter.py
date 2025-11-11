"""
Rep counter for bicep curls and hammer curls (both use same logic)
"""
from .base_counter import BaseRepCounter

class CurlCounter(BaseRepCounter):
    """Counter for bicep and hammer curls with improved accuracy"""
    
    def __init__(self):
        super().__init__()
        self.angle_history = []
        self.history_size = 5
        self.min_curl_angle = 60   # Must curl below 60 degrees
        self.min_extend_angle = 150  # Must extend above 150 degrees
        self.debounce_frames = 3
        self.frames_in_position = 0
        
    def count_rep(self, landmarks):
        """
        Count reps for curls (both bicep and hammer)
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
            if smoothed_angle > self.min_extend_angle:
                self.stage = "down"
                self.frames_in_position = 0
                    
            elif smoothed_angle < self.min_curl_angle:
                if self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                    self.frames_in_position = 0
            
            return self.counter, self.stage, int(smoothed_angle)
            
        except Exception as e:
            print(f"Error in curl counter: {e}")
            return self.counter, self.stage, None
