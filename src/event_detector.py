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


class InMemoryBaselineStore:
    """Baseline history kept in the process. Useful for tests and one-off runs.

    Detection normally runs as a scheduled task in a fresh process, so
    production uses the database-backed store instead.
    """

    def __init__(self):
        self._history = defaultdict(list)

    def get_keyword_baselines(self, category: str, keywords: List[str],
                              window_hours: int = 168) -> Dict[str, Dict]:
        baselines = {}

        for keyword in keywords:
            counts = self._history.get((category, keyword), [])
            if counts:
                baselines[keyword] = {
                    'baseline': float(np.median(counts)),
                    'observations': len(counts)
                }

        return baselines

    def record_keyword_observations(self, category: str, counts: Dict[str, int]) -> int:
        for keyword, count in counts.items():
            self._history[(category, keyword)].append(count)

        return len(counts)


class EventDetector:
    """Finds emerging events through spike detection and clustering"""

    def __init__(self,
                 spike_threshold: float = 2.0,
                 min_cluster_size: int = 3,
                 time_window_minutes: int = 60,
                 baseline_store=None,
                 baseline_window_hours: int = 168,
                 min_baseline_observations: int = 3):
        self.spike_threshold = spike_threshold
        self.min_cluster_size = min_cluster_size
        self.time_window = timedelta(minutes=time_window_minutes)

        self.baseline_store = baseline_store or InMemoryBaselineStore()
        self.baseline_window_hours = baseline_window_hours
        # A ratio against one or two past runs is noise, not a trend.
        self.min_baseline_observations = min_baseline_observations

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
            current_counts = dict(keywords.most_common(20))

            baselines = self.baseline_store.get_keyword_baselines(
                category, list(current_counts), self.baseline_window_hours
            )

            for keyword, count in current_counts.items():
                history = baselines.get(keyword)

                # No history means no evidence of a change, however loud the
                # keyword is right now.
                if not history or history['observations'] < self.min_baseline_observations:
                    continue

                baseline = history['baseline']
                if baseline <= 0:
                    continue

                spike_ratio = count / baseline

                if spike_ratio >= self.spike_threshold and count >= self.min_cluster_size:
                    spikes.append({
                        'type': 'keyword_spike',
                        'category': category,
                        'keyword': keyword,
                        'count': count,
                        'baseline': baseline,
                        'observations': history['observations'],
                        'spike_ratio': spike_ratio,
                        'severity': self._calculate_severity(spike_ratio),
                        'detected_at': datetime.utcnow().isoformat()
                    })

            # Recorded after comparison so this run's counts do not dilute the
            # baseline they were just measured against.
            self.baseline_store.record_keyword_observations(category, current_counts)

        logger.info(f"Detected {len(spikes)} keyword spikes")
        return spikes
    
    def detect_location_clusters(self, events: List[Dict]) -> List[Dict]:
        """Detect geographic clustering of events"""
        location_events = []
        
        # Extract events with locations
        for event in events:
            nlp_data = event.get('nlp_data', {})
            entities = nlp_data.get('entities', {})
            locations = entities.get('locations', [])
            
            if locations:
                location_events.append({
                    'event': event,
                    'locations': locations,
                    'category': event.get('category', 'unknown'),
                    'severity': nlp_data.get('severity_score', 0.0)
                })
        
        if not location_events:
            return []
        
        location_groups = defaultdict(list)
        
        for le in location_events:
            for location in le['locations']:
                location_groups[location].append(le)
        
        clusters = []
        
        for location, group in location_groups.items():
            if len(group) >= self.min_cluster_size:
                avg_severity = np.mean([g['severity'] for g in group])
                
                categories = [g['category'] for g in group]
                most_common_category = Counter(categories).most_common(1)[0][0]
                
                clusters.append({
                    'type': 'location_cluster',
                    'location': location,
                    'event_count': len(group),
                    'category': most_common_category,
                    'avg_severity': float(avg_severity),
                    'severity': self._calculate_severity(avg_severity * 2),
                    'detected_at': datetime.utcnow().isoformat()
                })
        
        logger.info(f"Detected {len(clusters)} location clusters")
        return clusters
    
    def detect_topic_clusters(self, events: List[Dict]) -> List[Dict]:
        """Detect similar events using text clustering"""
        if len(events) < self.min_cluster_size:
            return []
        
        texts = []
        valid_events = []
        
        for event in events:
            text = f"{event.get('title', '')} {event.get('description', '')}"
            if text.strip():
                texts.append(text)
                valid_events.append(event)
        
        if len(texts) < self.min_cluster_size:
            return []
        
        try:
            # Vectorize text
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Convert to distance matrix
            distance_matrix = 1 - similarity_matrix
            
            # Cluster using DBSCAN
            clustering = DBSCAN(
                eps=0.5,
                min_samples=self.min_cluster_size,
                metric='precomputed'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # Extract clusters
            clusters = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Noise
                    continue
                
                # Get cluster members
                cluster_indices = np.where(labels == label)[0]
                cluster_events = [valid_events[i] for i in cluster_indices]
                
                # Get cluster statistics
                categories = [e.get('category', 'unknown') for e in cluster_events]
                most_common_category = Counter(categories).most_common(1)[0][0]
                
                # Calculate severity
                severities = []
                for e in cluster_events:
                    nlp_data = e.get('nlp_data', {})
                    severities.append(nlp_data.get('severity_score', 0.0))
                avg_severity = np.mean(severities)
                
                # Extract common keywords
                cluster_texts = [texts[i] for i in cluster_indices]
                keywords = self._extract_cluster_keywords(cluster_texts, vectorizer)
                
                clusters.append({
                    'type': 'topic_cluster',
                    'cluster_id': f'cluster_{label}',
                    'event_count': len(cluster_events),
                    'category': most_common_category,
                    'keywords': keywords[:5],
                    'avg_severity': float(avg_severity),
                    'severity': self._calculate_severity(avg_severity * 2),
                    'detected_at': datetime.utcnow().isoformat()
                })
            
            logger.info(f"Detected {len(clusters)} topic clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Topic clustering failed: {e}")
            return []
    
    def _extract_cluster_keywords(self, texts: List[str], 
                                  vectorizer: TfidfVectorizer) -> List[str]:
        """Extract top keywords from cluster"""
        try:
            tfidf_matrix = vectorizer.transform(texts)
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            
            feature_names = vectorizer.get_feature_names_out()
            top_indices = mean_scores.argsort()[-10:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except:
            return []
    
    def detect_all_events(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Run all detection methods"""
        logger.info(f"Running event detection on {len(events)} events")
        
        detected_events = {
            'keyword_spikes': self.detect_keyword_spikes(events),
            'location_clusters': self.detect_location_clusters(events),
            'topic_clusters': self.detect_topic_clusters(events)
        }
        
        total_detected = sum(len(v) for v in detected_events.values())
        logger.info(f"Total detected events: {total_detected}")
        
        return detected_events
    
    def _calculate_severity(self, value: float) -> str:
        """Calculate severity level"""
        if value >= 5.0:
            return 'critical'
        elif value >= 3.0:
            return 'high'
        elif value >= 2.0:
            return 'medium'
        else:
            return 'low'
    
    def filter_high_priority(self, detected_events: Dict[str, List[Dict]], 
                            min_severity: str = 'medium') -> List[Dict]:
        """Filter for high-priority events"""
        severity_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_level = severity_levels.get(min_severity, 1)
        
        priority_events = []
        
        for event_type, events in detected_events.items():
            for event in events:
                severity = event.get('severity', 'low')
                if severity_levels.get(severity, 0) >= min_level:
                    priority_events.append(event)
        
        # Sort by severity
        priority_events.sort(
            key=lambda x: severity_levels.get(x.get('severity', 'low'), 0),
            reverse=True
        )
        
        return priority_events


if __name__ == '__main__':
    # quick test
    detector = EventDetector(spike_threshold=2.0, min_cluster_size=3)
    
    test_events = [
        {
            'title': 'Earthquake hits California',
            'description': 'Major seismic activity reported',
            'category': 'natural_disaster',
            'nlp_data': {
                'entities': {'locations': ['California']},
                'severity_score': 0.8
            }
        },
        {
            'title': 'California earthquake aftermath',
            'description': 'Damage assessment underway',
            'category': 'natural_disaster',
            'nlp_data': {
                'entities': {'locations': ['California']},
                'severity_score': 0.7
            }
        },
        {
            'title': 'Earthquake emergency response',
            'description': 'Emergency services mobilized',
            'category': 'natural_disaster',
            'nlp_data': {
                'entities': {'locations': ['California']},
                'severity_score': 0.9
            }
        }
    ]
    
    detected = detector.detect_all_events(test_events)
    print("Detected events:", detected)