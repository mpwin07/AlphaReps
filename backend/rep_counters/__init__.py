"""
Rep counters module for different exercises
"""
from .base_counter import BaseRepCounter
from .curl_counter import CurlCounter
from .squat_counter import SquatCounter
from .pushup_counter import PushupCounter
from .shoulder_press_counter import ShoulderPressCounter

__all__ = [
    'BaseRepCounter',
    'CurlCounter',
    'SquatCounter',
    'PushupCounter',
    'ShoulderPressCounter'
]
