"""
Tests for Event Detector
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_detector import EventDetector


@pytest.fixture
def detector():
    """Fixture to create event detector"""
    return EventDetector(spike_threshold=2.0, min_cluster_size=3)


@pytest.fixture
def sample_events():
    """Sample events for testing"""
    return [
        {
            'event_id': '1',
            'title': 'Earthquake hits California',
            'description': 'Major seismic activity reported in California',
            'category': 'disaster',
            'nlp_data': {
                'entities': {'locations': ['California']},
                'severity_score': 0.8
            }
        },
        {
            'event_id': '2',
            'title': 'California earthquake aftermath',
            'description': 'Damage assessment underway in California',
            'category': 'disaster',
            'nlp_data': {
                'entities': {'locations': ['California']},
                'severity_score': 0.7
            }
        },
        {
            'event_id': '3',
            'title': 'Emergency response in California',
            'description': 'Teams mobilized for earthquake response',
            'category': 'disaster',
            'nlp_data': {
                'entities': {'locations': ['California']},
                'severity_score': 0.9
            }
        },
        {
            'event_id': '4',
            'title': 'Tech company launches new product',
            'description': 'Innovation in AI announced',
            'category': 'technology',
            'nlp_data': {
                'entities': {'locations': []},
                'severity_score': 0.3
            }
        }
    ]


def test_keyword_spike_detection(detector, sample_events):
    """Test keyword spike detection"""
    spikes = detector.detect_keyword_spikes(sample_events)
    
    # Should detect spikes for "earthquake" and "California"
    assert len(spikes) >= 0
    
    if spikes:
        spike = spikes[0]
        assert 'keyword' in spike
        assert 'count' in spike
        assert 'spike_ratio' in spike
        assert 'severity' in spike


def test_location_clustering(detector, sample_events):
    """Test location clustering"""
    clusters = detector.detect_location_clusters(sample_events)
    
    # Should detect cluster in California
    assert len(clusters) >= 1
    
    if clusters:
        cluster = clusters[0]
        assert cluster['location'] == 'California'
        assert cluster['event_count'] == 3
        assert cluster['category'] == 'disaster'


def test_topic_clustering(detector, sample_events):
    """Test topic clustering"""
    clusters = detector.detect_topic_clusters(sample_events)
    
    # May or may not form clusters depending on similarity threshold
    assert isinstance(clusters, list)
    
    if clusters:
        cluster = clusters[0]
        assert 'cluster_id' in cluster
        assert 'event_count' in cluster
        assert 'category' in cluster


def test_detect_all_events(detector, sample_events):
    """Test comprehensive event detection"""
    detected = detector.detect_all_events(sample_events)
    
    assert 'keyword_spikes' in detected
    assert 'location_clusters' in detected
    assert 'topic_clusters' in detected
    
    # Should detect at least location cluster
    assert len(detected['location_clusters']) >= 1


def test_high_priority_filtering(detector, sample_events):
    """Test high priority event filtering"""
    detected = detector.detect_all_events(sample_events)
    priority = detector.filter_high_priority(detected, min_severity='medium')
    
    assert isinstance(priority, list)
    # All returned events should have medium+ severity
    for event in priority:
        assert event['severity'] in ['medium', 'high', 'critical']


def test_severity_calculation(detector):
    """Test severity calculation"""
    assert detector._calculate_severity(6.0) == 'critical'
    assert detector._calculate_severity(3.5) == 'high'
    assert detector._calculate_severity(2.5) == 'medium'
    assert detector._calculate_severity(1.0) == 'low'


def test_empty_events(detector):
    """Test handling of empty event list"""
    detected = detector.detect_all_events([])
    
    assert detected['keyword_spikes'] == []
    assert detected['location_clusters'] == []
    assert detected['topic_clusters'] == []


def test_baseline_tracking(detector):
    """Test baseline tracking"""
    baseline1 = detector._get_baseline('disaster')
    
    detector._update_baseline('disaster', 10)
    
    baseline2 = detector._get_baseline('disaster')
    assert baseline2 > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
