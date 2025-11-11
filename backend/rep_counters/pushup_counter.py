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
        # Smoothing for better accuracy
        self.back_angle_history = []
        self.elbow_angle_history = []
        self.history_size = 5
        
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
            current_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            
            # Calculate average back angle (posture)
            left_back_angle = self.calculate_angle(l_shoulder, l_hip, l_ankle)
            right_back_angle = self.calculate_angle(r_shoulder, r_hip, r_ankle)
            current_back_angle = (left_back_angle + right_back_angle) / 2
            
            # Smooth angles using moving average for better accuracy
            self.elbow_angle_history.append(current_elbow_angle)
            if len(self.elbow_angle_history) > self.history_size:
                self.elbow_angle_history.pop(0)
            self.avg_elbow_angle = sum(self.elbow_angle_history) / len(self.elbow_angle_history)
            
            self.back_angle_history.append(current_back_angle)
            if len(self.back_angle_history) > self.history_size:
                self.back_angle_history.pop(0)
            self.avg_back_angle = sum(self.back_angle_history) / len(self.back_angle_history)
            
            # Check elbow flare (elbows should be tucked)
            left_elbow_flare = self.calculate_angle(l_hip, l_shoulder, l_elbow)
            right_elbow_flare = self.calculate_angle(r_hip, r_shoulder, r_elbow)
            
            # Posture feedback with stricter thresholds
            feedback_issues = []
            
            # 1. Check back alignment (should be straight: 150-180 degrees for better form)
            if self.avg_back_angle < 150:
                if self.avg_back_angle < 140:
                    feedback_issues.append("STRAIGHTEN BACK - HIPS SAGGING")
                else:
                    feedback_issues.append("STRAIGHTEN BACK")
            
            # 2. Check elbow flare (should be < 60 degrees for proper form - stricter)
            avg_elbow_flare = (left_elbow_flare + right_elbow_flare) / 2
            if avg_elbow_flare > 60:
                if avg_elbow_flare > 70:
                    feedback_issues.append("TUCK ELBOWS IN MORE")
                else:
                    feedback_issues.append("TUCK ELBOWS")
            
            # 3. Check if hips are too high (pike position)
            if self.avg_back_angle > 175:
                feedback_issues.append("LOWER HIPS")
            
            # 4. Check elbow symmetry
            elbow_diff = abs(left_elbow_angle - right_elbow_angle)
            if elbow_diff > 15:
                feedback_issues.append("BALANCE BOTH ARMS")
            
            # Provide feedback
            if len(feedback_issues) > 0:
                form_feedback = " | ".join(feedback_issues[:2])  # Show max 2 issues
                self.form_quality = "POOR"
            else:
                form_feedback = "PERFECT FORM"
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
                # Check if in proper starting position (stricter requirements)
                if is_body_visible and self.avg_back_angle > 150 and self.avg_elbow_angle > 160:
                    self.state = 'ready'
                    self.feedback = form_feedback
                    self.stage = "up"
                else:
                    self.feedback = "GET INTO PLANK POSITION"
            
            elif self.state == 'ready' or self.stage == "up":
                self.feedback = form_feedback
                # Going down (elbow bending) - must go below 95 degrees
                if self.avg_elbow_angle < 95:
                    self.stage = "down"
            
            elif self.stage == "down":
                self.feedback = form_feedback
                # Going up (elbow extending) - must extend above 160 degrees
                if self.avg_elbow_angle > 160:
                    # Only count rep if form was good throughout
                    if not has_form_issues:
                        self.counter += 1
                        self.feedback = "✓ REP COUNTED - EXCELLENT!"
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
