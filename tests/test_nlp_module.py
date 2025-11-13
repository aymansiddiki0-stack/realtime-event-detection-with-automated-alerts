"""
Tests for NLP Module
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nlp_module import NLPProcessor


@pytest.fixture
def nlp_processor():
    """Fixture to create NLP processor"""
    return NLPProcessor()


def test_clean_text(nlp_processor):
    """Test text cleaning"""
    dirty_text = "Check this out! http://example.com and email me@example.com   "
    clean = nlp_processor.clean_text(dirty_text)
    
    assert "http" not in clean
    assert "@" not in clean
    assert clean == "Check this out and email"


def test_extract_entities(nlp_processor):
    """Test entity extraction"""
    text = "President Biden met with officials in Washington on Monday."
    entities = nlp_processor.extract_entities(text)
    
    assert 'persons' in entities
    assert 'locations' in entities
    assert 'dates' in entities
    assert len(entities['persons']) > 0 or len(entities['organizations']) > 0


def test_classify_event(nlp_processor):
    """Test event classification"""
    text = "A major earthquake struck California, causing widespread damage."
    category, confidence = nlp_processor.classify_event(text)
    
    assert category in ['natural disaster', 'disaster', 'other']
    assert 0 <= confidence <= 1


def test_crisis_level_detection(nlp_processor):
    """Test crisis level detection"""
    text = "Critical emergency: Multiple casualties reported in major disaster."
    entities = {'locations': ['California'], 'persons': []}
    
    level, score = nlp_processor.detect_crisis_level(text, entities)
    
    assert level in ['low', 'medium', 'high', 'critical']
    assert 0 <= score <= 1
    assert level in ['high', 'critical']  # Should be high severity


def test_process_event(nlp_processor):
    """Test full event processing"""
    event = {
        'event_id': 'test_1',
        'title': 'Earthquake strikes California',
        'description': 'A major seismic event occurred',
        'content': 'Emergency services are responding'
    }
    
    processed = nlp_processor.process_event(event)
    
    assert 'nlp_data' in processed
    assert 'category' in processed['nlp_data']
    assert 'entities' in processed['nlp_data']
    assert 'crisis_level' in processed['nlp_data']


def test_empty_text(nlp_processor):
    """Test handling of empty text"""
    entities = nlp_processor.extract_entities("")
    
    assert entities['persons'] == []
    assert entities['locations'] == []


def test_batch_processing(nlp_processor):
    """Test batch processing"""
    events = [
        {'title': 'Event 1', 'description': 'Description 1'},
        {'title': 'Event 2', 'description': 'Description 2'}
    ]
    
    processed = nlp_processor.batch_process(events)
    
    assert len(processed) == 2
    assert all('nlp_data' in e for e in processed)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
