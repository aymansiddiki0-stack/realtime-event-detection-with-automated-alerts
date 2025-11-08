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

