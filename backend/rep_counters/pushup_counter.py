"""
Enhanced Rep counter for push-ups with posture correction
"""
from .base_counter import BaseRepCounter

class PushupCounter(BaseRepCounter):
    """Enhanced counter for push-ups with posture correction"""
    
    def __init__(self):
        super().__init__()
        self.state = 'get_ready'
        self.feedback = ''
        self.visibility_threshold = 0.8
        self.avg_back_angle = 0
        self.avg_elbow_angle = 0
        self.form_quality = "GOOD"
        
    def check_posture(self, landmarks):
        """
        Check push-up posture and provide feedback
        Returns: (form_feedback, back_angle, elbow_flare_issue)
        """
        try:
            # Get left side landmarks
            l_shoulder = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            l_elbow = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
            ]
            l_wrist = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
            ]
            l_hip = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            l_ankle = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ]
            
            # Get right side landmarks
            r_shoulder = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            r_elbow = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]
            r_wrist = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            ]
            r_hip = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            r_ankle = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]
            
            # Calculate average elbow angle (arms)
            left_elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            right_elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            self.avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            
            # Calculate average back angle (posture)
            left_back_angle = self.calculate_angle(l_shoulder, l_hip, l_ankle)
            right_back_angle = self.calculate_angle(r_shoulder, r_hip, r_ankle)
            self.avg_back_angle = (left_back_angle + right_back_angle) / 2
            
            # Check elbow flare (elbows should be tucked)
            left_elbow_flare = self.calculate_angle(l_hip, l_shoulder, l_elbow)
            right_elbow_flare = self.calculate_angle(r_hip, r_shoulder, r_elbow)
            
            # Posture feedback
            feedback_issues = []
            
            # 1. Check back alignment (should be straight: 145-180 degrees)
            if self.avg_back_angle < 145:
                feedback_issues.append("STRAIGHTEN BACK")
            
            # 2. Check elbow flare (should be < 65 degrees for proper form)
            if left_elbow_flare > 65 or right_elbow_flare > 65:
                feedback_issues.append("TUCK ELBOWS")
            
            # 3. Check if hips are sagging
            if self.avg_back_angle < 135:
                feedback_issues.append("HIPS TOO LOW")
            
            # Provide feedback
            if len(feedback_issues) > 0:
                form_feedback = " | ".join(feedback_issues)
                self.form_quality = "POOR"
            else:
                form_feedback = "GOOD FORM"
                self.form_quality = "GOOD"
            
            return form_feedback, self.avg_back_angle, len(feedback_issues) > 0
            
        except Exception as e:
            print(f"Error checking posture: {e}")
            return "CHECK FORM", 0, False
    
    def count_rep(self, landmarks):
        """
        Count reps for push-ups with posture correction
        Returns: (rep_count, stage, angle, feedback, back_angle, form_quality)
        """
        try:
            # Check visibility of key landmarks
            l_shoulder_val = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_elbow_val = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            l_hip_val = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            r_shoulder_val = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_elbow_val = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            r_hip_val = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            is_body_visible = all(lm.visibility > self.visibility_threshold for lm in 
                                [l_shoulder_val, l_elbow_val, l_hip_val, 
                                 r_shoulder_val, r_elbow_val, r_hip_val])
            
            # Get posture feedback
            form_feedback, back_angle, has_form_issues = self.check_posture(landmarks)
            
            # State machine for rep counting with posture awareness
            if self.state == 'get_ready':
                # Check if in proper starting position
                if is_body_visible and self.avg_back_angle > 145 and self.avg_elbow_angle > 155:
                    self.state = 'ready'
                    self.feedback = form_feedback
                    self.stage = "up"
                else:
                    self.feedback = "GET INTO PLANK POSITION"
            
            elif self.state == 'ready' or self.stage == "up":
                self.feedback = form_feedback
                # Going down (elbow bending)
                if self.avg_elbow_angle < 90:
                    self.stage = "down"
            
            elif self.stage == "down":
                self.feedback = form_feedback
                # Going up (elbow extending)
                if self.avg_elbow_angle > 155:
                    # Only count rep if form was good
                    if not has_form_issues:
                        self.counter += 1
                        self.feedback = "REP COUNTED!"
                    else:
                        self.feedback = form_feedback + " | REP NOT COUNTED"
                    
                    self.stage = "up"
                    self.state = 'ready'
            
            return (self.counter, self.stage, self.avg_elbow_angle, 
                   self.feedback, self.avg_back_angle, self.form_quality)
            
        except Exception as e:
            print(f"Error in enhanced pushup counter: {e}")
            self.state = 'get_ready'
            self.feedback = "NO BODY DETECTED"
            return (self.counter, self.stage, None, 
                   self.feedback, 0, "POOR")
