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
