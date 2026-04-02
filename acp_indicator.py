"""
Auto Chart Patterns (ACP) Indicator - Python Implementation
Converted from TradingView Pine Script (Trendoscope®)
Detects geometric price patterns: Channels, Wedges, Triangles
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum

class PatternType(Enum):
    ASCENDING_CHANNEL = "Ascending Channel"
    DESCENDING_CHANNEL = "Descending Channel"
    RANGING_CHANNEL = "Ranging Channel"
    RISING_WEDGE_EXP = "Rising Wedge (Expanding)"
    FALLING_WEDGE_EXP = "Falling Wedge (Expanding)"
    RISING_WEDGE_CON = "Rising Wedge (Contracting)"
    FALLING_WEDGE_CON = "Falling Wedge (Contracting)"
    ASCENDING_TRIANGLE_EXP = "Ascending Triangle (Expanding)"
    DESCENDING_TRIANGLE_EXP = "Descending Triangle (Expanding)"
    CONVERGING_TRIANGLE = "Converging Triangle"
    ASCENDING_TRIANGLE_CON = "Ascending Triangle (Contracting)"
    DESCENDING_TRIANGLE_CON = "Descending Triangle (Contracting)"
    DIVERGING_TRIANGLE = "Diverging Triangle"

@dataclass
class Pivot:
    """Represents a pivot point in the zigzag"""
    index: int
    price: float
    direction: int  # 1 for high, -1 for low

@dataclass
class TrendLine:
    """Represents a trend line between two pivots"""
    start_pivot: Pivot
    end_pivot: Pivot
    slope: float
    intercept: float

@dataclass
class Pattern:
    """Detected chart pattern"""
    pattern_type: PatternType
    start_index: int
    end_index: int
    pivots: List[Pivot]
    upper_line: TrendLine
    lower_line: TrendLine
    confidence: float

class ZigzagCalculator:
    """Calculates zigzag levels from high/low prices"""
    
    def __init__(self, length: int, depth: int):
        self.length = length
        self.depth = depth
        self.pivots: List[Pivot] = []
        
    def calculate(self, highs: np.ndarray, lows: np.ndarray) -> List[Pivot]:
        """Calculate zigzag pivots"""
        self.pivots = []
        
        if len(highs) < self.length:
            return self.pivots
            
        n = len(highs)
        pivot_types = np.zeros(n)  # 0=none, 1=high, -1=low
        
        # Find initial pivot
        window_high = np.max(highs[:self.length])
        window_low = np.min(lows[:self.length])
        
        last_pivot_type = 0
        last_pivot_price = 0
        last_pivot_index = 0
        
        for i in range(self.length, n):
            # Get window max/min
            start_idx = max(0, i - self.length)
            window_high = np.max(highs[start_idx:i+1])
            window_low = np.min(lows[start_idx:i+1])
            
            # Check for high pivot
            if highs[i] == window_high and last_pivot_type != 1:
                if last_pivot_type == 0 or (highs[i] - last_pivot_price) / last_pivot_price > self.depth / 100:
                    pivot_types[i] = 1
                    last_pivot_type = 1
                    last_pivot_price = highs[i]
                    last_pivot_index = i
                    self.pivots.append(Pivot(i, highs[i], 1))
            
            # Check for low pivot
            elif lows[i] == window_low and last_pivot_type != -1:
                if last_pivot_type == 0 or (last_pivot_price - lows[i]) / last_pivot_price > self.depth / 100:
                    pivot_types[i] = -1
                    last_pivot_type = -1
                    last_pivot_price = lows[i]
                    last_pivot_index = i
                    self.pivots.append(Pivot(i, lows[i], -1))
        
        return self.pivots

class PatternDetector:
    """Detects chart patterns from zigzag pivots"""
    
    def __init__(self, error_threshold: float = 0.20, flat_threshold: float = 0.20):
        self.error_threshold = error_threshold
        self.flat_threshold = flat_threshold
        
    def fit_trendline(self, p1: Pivot, p2: Pivot) -> TrendLine:
        """Fit trend line between two pivots"""
        if p1.index == p2.index:
            return None
        
        slope = (p2.price - p1.price) / (p2.index - p1.index)
        intercept = p1.price - slope * p1.index
        
        return TrendLine(p1, p2, slope, intercept)
    
    def get_trendline_price(self, trendline: TrendLine, index: int) -> float:
        """Get price on trend line at given index"""
        return trendline.slope * index + trendline.intercept
    
    def validate_trendline(self, trendline: TrendLine, prices: np.ndarray, 
                          start_idx: int, end_idx: int) -> float:
        """Validate trend line and return error ratio"""
        if trendline is None:
            return 999
            
        max_error = 0
        for i in range(start_idx, end_idx + 1):
            expected_price = self.get_trendline_price(trendline, i)
            error_ratio = abs(prices[i] - expected_price) / expected_price
            max_error = max(max_error, error_ratio)
        
        return max_error
    
    def is_flat(self, trendline: TrendLine) -> bool:
        """Check if trend line is flat"""
        return abs(trendline.slope) < self.flat_threshold
    
    def detect_patterns(self, pivots: List[Pivot], highs: np.ndarray, 
                       lows: np.ndarray, num_pivots: int = 5) -> List[Pattern]:
        """Detect patterns from zigzag pivots"""
        patterns = []
        
        if len(pivots) < num_pivots:
            return patterns
        
        prices = (highs + lows) / 2
        
        # Check for 5-pivot patterns
        for i in range(len(pivots) - num_pivots + 1):
            pivot_set = pivots[i:i+num_pivots]
            
            # Try to fit upper and lower trend lines
            upper_line = self.fit_trendline(pivot_set[0], pivot_set[-1])
            lower_line = self.fit_trendline(pivot_set[1], pivot_set[-2])
            
            if not upper_line or not lower_line:
                continue
            
            # Validate lines
            start_idx = pivot_set[0].index
            end_idx = pivot_set[-1].index
            
            error_upper = self.validate_trendline(upper_line, prices, start_idx, end_idx)
            error_lower = self.validate_trendline(lower_line, prices, start_idx, end_idx)
            
            # Check if pattern is valid
            if error_upper < self.error_threshold and error_lower < self.error_threshold:
                pattern_type = self._classify_pattern(pivot_set, upper_line, lower_line)
                if pattern_type:
                    confidence = 1.0 - max(error_upper, error_lower)
                    patterns.append(Pattern(
                        pattern_type=pattern_type,
                        start_index=start_idx,
                        end_index=end_idx,
                        pivots=pivot_set,
                        upper_line=upper_line,
                        lower_line=lower_line,
                        confidence=confidence
                    ))
        
        return patterns
    
    def _classify_pattern(self, pivots: List[Pivot], upper: TrendLine, 
                         lower: TrendLine) -> PatternType:
        """Classify pattern type based on trend lines"""
        upper_flat = self.is_flat(upper)
        lower_flat = self.is_flat(lower)
        
        upper_up = upper.slope > 0
        lower_up = lower.slope > 0
        
        # Diverging (slope differences increasing)
        upper_dist_start = upper.end_pivot.price - upper.start_pivot.price
        lower_dist_start = lower.start_pivot.price - lower.end_pivot.price
        
        diverging = upper_dist_start > 0 and lower_dist_start > 0
        converging = upper_dist_start < 0 and lower_dist_start < 0
        
        # Channel patterns
        if upper_flat and lower_flat:
            return PatternType.RANGING_CHANNEL
        elif upper_up and lower_up:
            return PatternType.ASCENDING_CHANNEL if not diverging else None
        elif not upper_up and not lower_up:
            return PatternType.DESCENDING_CHANNEL if not diverging else None
        
        # Wedge patterns
        elif upper_up and lower_up:
            return PatternType.RISING_WEDGE_EXP if diverging else PatternType.RISING_WEDGE_CON
        elif not upper_up and not lower_up:
            return PatternType.FALLING_WEDGE_EXP if diverging else PatternType.FALLING_WEDGE_CON
        
        # Triangle patterns
        elif (upper_up and not lower_up) or (not upper_up and lower_up):
            if upper_up and not lower_up:
                return PatternType.ASCENDING_TRIANGLE_EXP if diverging else PatternType.ASCENDING_TRIANGLE_CON
            else:
                return PatternType.DESCENDING_TRIANGLE_EXP if diverging else PatternType.DESCENDING_TRIANGLE_CON
        elif upper_flat or lower_flat:
            if converging:
                return PatternType.CONVERGING_TRIANGLE
            elif diverging:
                return PatternType.DIVERGING_TRIANGLE
        
        return None

class ACPIndicator:
    """Auto Chart Patterns Indicator"""
    
    def __init__(self, zigzag_length: int = 8, depth: int = 55, 
                 num_pivots: int = 5, error_threshold: float = 0.20):
        self.zigzag = ZigzagCalculator(zigzag_length, depth)
        self.detector = PatternDetector(error_threshold)
        self.num_pivots = num_pivots
        self.patterns: List[Pattern] = []
    
    def analyze(self, df: pd.DataFrame) -> List[Pattern]:
        """Analyze price data and detect patterns"""
        if len(df) < 20:
            return []
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Calculate zigzag
        pivots = self.zigzag.calculate(highs, lows)
        
        if len(pivots) < self.num_pivots:
            return []
        
        # Detect patterns
        self.patterns = self.detector.detect_patterns(pivots, highs, lows, self.num_pivots)
        
        return self.patterns
    
    def get_latest_patterns(self, top_n: int = 3) -> List[Pattern]:
        """Get top recent patterns by confidence"""
        return sorted(self.patterns, key=lambda p: p.confidence, reverse=True)[:top_n]
