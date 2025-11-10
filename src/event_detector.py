"""
Event detection - finds spikes, clusters, and trending topics
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventDetector:
    """Finds emerging events through spike detection and clustering"""

    def __init__(self,
                 spike_threshold: float = 2.0,
                 min_cluster_size: int = 3,
                 time_window_minutes: int = 60):
        self.spike_threshold = spike_threshold
        self.min_cluster_size = min_cluster_size
        self.time_window = timedelta(minutes=time_window_minutes)

        # keep track of historical rates for baseline comparison
        self.historical_rates = defaultdict(list)
        self.baseline_window = 24 * 60

        logger.info(f"Event detector initialized (threshold={spike_threshold}, "
                   f"min_cluster={min_cluster_size}, window={time_window_minutes}m)")
    
    def detect_keyword_spikes(self, events: List[Dict]) -> List[Dict]:
        """Detects spikes in keyword mentions"""
        if not events:
            return []
        
        category_keywords = defaultdict(Counter)
        
        for event in events:
            category = event.get('category', 'unknown')
            text = f"{event.get('title', '')} {event.get('description', '')}"
            words = text.lower().split()
            
            # Count significant words (length > 3)
            significant_words = [w for w in words if len(w) > 3]
            category_keywords[category].update(significant_words)
        
        # Detects spikes
        spikes = []
        
        for category, keywords in category_keywords.items():
            # Get historical baseline
            baseline = self._get_baseline(category)
            
            # Check each keyword
            for keyword, count in keywords.most_common(20):
                if baseline > 0:
                    spike_ratio = count / baseline
                    
                    if spike_ratio >= self.spike_threshold and count >= self.min_cluster_size:
                        spikes.append({
                            'type': 'keyword_spike',
                            'category': category,
                            'keyword': keyword,
                            'count': count,
                            'baseline': baseline,
                            'spike_ratio': spike_ratio,
                            'severity': self._calculate_severity(spike_ratio),
                            'detected_at': datetime.utcnow().isoformat()
                        })
            
            # Update baseline
            self._update_baseline(category, len(keywords))
        
        logger.info(f"Detected {len(spikes)} keyword spikes")
        return spikes
