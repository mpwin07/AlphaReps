"""
Base class for exercise rep counters
"""
import numpy as np
import mediapipe as mp
from abc import ABC, abstractmethod

class BaseRepCounter(ABC):
    """Abstract base class for rep counters"""
    
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    @abstractmethod
    def count_rep(self, landmarks):
        """Count reps based on landmarks - must be implemented by subclasses"""
        pass
    
    def reset(self):
        """Reset counter"""
        self.counter = 0
        self.stage = None
    
    def get_count(self):
        """Get current count"""
        return self.counter
    
    def get_stage(self):
        """Get current stage"""
        return self.stage
