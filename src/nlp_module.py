"""
NLP processing - entity extraction, classification, crisis detection
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import spacy
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPProcessor:
    """Main NLP pipeline for processing events"""

    def __init__(self):
        # Load spaCy for entity extraction
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded spaCy model successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

        # Zero-shot classifier for categorization
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli"
            )
            logger.info("Loaded classification model successfully")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")
            self.classifier = None

        self.categories = [
            'natural disaster',
            'political event',
            'conflict/violence',
            'technology',
            'health/pandemic',
            'economy/business',
            'climate/environment',
            'terrorism/security',
            'sports',
            'entertainment'
        ]

        # Keywords for crisis detection
        self.crisis_keywords = {
            'disaster': [
                'earthquake', 'tsunami', 'hurricane', 'tornado', 'flood',
                'wildfire', 'drought', 'avalanche', 'volcano', 'landslide'
            ],
            'conflict': [
                'war', 'attack', 'shooting', 'bombing', 'military',
                'violence', 'protest', 'riot', 'terrorism', 'conflict'
            ],
            'health': [
                'outbreak', 'pandemic', 'epidemic', 'virus', 'disease',
                'infection', 'hospital', 'death', 'emergency'
            ],
            'political': [
                'election', 'vote', 'parliament', 'president', 'minister',
                'government', 'policy', 'legislation', 'democracy'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Strip URLs, emails, extra whitespace"""
        if not text:
            return ""

        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)

        return text

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Pull out people, places, orgs using spaCy"""
        if not text:
            return {
                'persons': [],
                'organizations': [],
                'locations': [],
                'dates': [],
                'other': []
            }
        
        doc = self.nlp(text[:10000])  # cap at 10k chars for speed

        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'other': []
        }

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ in ['ORG', 'NORP']:
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC', 'FAC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            else:
                entities['other'].append(ent.text)

        # dedupe
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def classify_event(self, text: str) -> Tuple[str, float]:
        """Categorize the event using zero-shot classification"""
        if not text or not self.classifier:
            return self._keyword_classify(text)
        
        try:
            result = self.classifier(
                text[:256],  # keep it short for speed
                self.categories,
                multi_label=False
            )

            category = result['labels'][0]
            confidence = result['scores'][0]

            return category, confidence

        except Exception as e:
            logger.warning(f"Classification failed, using keyword fallback: {e}")
            return self._keyword_classify(text)

    def _keyword_classify(self, text: str) -> Tuple[str, float]:
        """Simple keyword matching as fallback"""
        if not text:
            return 'other', 0.0
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.crisis_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        if not scores or max(scores.values()) == 0:
            return 'other', 0.3

        top_category = max(scores, key=scores.get)
        max_score = scores[top_category]

        confidence = min(max_score / 5.0, 1.0)

        return top_category, confidence

    def detect_crisis_level(self, text: str, entities: Dict) -> Tuple[str, float]:
        """Figure out how severe the event is"""
        text_lower = text.lower() if text else ""

        high_priority = [
            'dead', 'killed', 'deaths', 'casualties', 'emergency',
            'urgent', 'critical', 'severe', 'major', 'massive'
        ]

        medium_priority = [
            'injured', 'damage', 'warning', 'alert', 'concern',
            'significant', 'serious', 'considerable'
        ]

        high_count = sum(1 for kw in high_priority if kw in text_lower)
        medium_count = sum(1 for kw in medium_priority if kw in text_lower)

        severity_score = (high_count * 3 + medium_count * 1.5) / 10.0

        # more locations = wider impact
        location_factor = min(len(entities.get('locations', [])) * 0.1, 0.3)
        severity_score += location_factor

        if severity_score >= 0.7:
            level = 'critical'
        elif severity_score >= 0.4:
            level = 'high'
        elif severity_score >= 0.2:
            level = 'medium'
        else:
            level = 'low'

        return level, min(severity_score, 1.0)

    def extract_keywords(self, texts: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """TF-IDF keyword extraction"""
        if not texts:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            top_indices = mean_scores.argsort()[-top_n:][::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]

            return keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    def process_event(self, event: Dict) -> Dict:
        """Run full NLP pipeline on a single event"""
        text = self._get_text_content(event)

        cleaned_text = self.clean_text(text)

        entities = self.extract_entities(cleaned_text)

        category, confidence = self.classify_event(cleaned_text)

        crisis_level, severity_score = self.detect_crisis_level(cleaned_text, entities)

        event['nlp_data'] = {
            'cleaned_text': cleaned_text[:1000],
            'entities': entities,
            'category': category,
            'category_confidence': float(confidence),
            'crisis_level': crisis_level,
            'severity_score': float(severity_score),
            'word_count': len(cleaned_text.split())
        }

        return event

    def _get_text_content(self, event: Dict) -> str:
        """Combine title, description, content into one string"""
        text_parts = []

        if event.get('title'):
            text_parts.append(event['title'])

        if event.get('description'):
            text_parts.append(event['description'])

        if event.get('content'):
            text_parts.append(event['content'])

        return ' '.join(text_parts)

    def batch_process(self, events: List[Dict]) -> List[Dict]:
        """Process a batch of events"""
        processed_events = []

        for event in events:
            try:
                processed_event = self.process_event(event)
                processed_events.append(processed_event)
            except Exception as e:
                logger.error(f"Failed to process event {event.get('event_id')}: {e}")
                # still add it with minimal data
                event['nlp_data'] = {
                    'error': str(e),
                    'category': 'unknown',
                    'category_confidence': 0.0,
                    'crisis_level': 'low',
                    'severity_score': 0.0
                }
                processed_events.append(event)

        return processed_events


# singleton pattern
_nlp_processor = None


def get_nlp_processor() -> NLPProcessor:
    """Get the NLP processor instance"""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor()
    return _nlp_processor


if __name__ == '__main__':
    # quick test
    processor = NLPProcessor()

    test_event = {
        'title': 'Major earthquake hits California coast',
        'description': 'A 7.5 magnitude earthquake struck near Los Angeles, causing significant damage',
        'content': 'Emergency services are responding to a major seismic event in Southern California...'
    }

    processed = processor.process_event(test_event)
    print("Processed event:", processed['nlp_data'])