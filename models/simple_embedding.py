import os
import numpy as np
from typing import List, Union
import logging
from config import Config

class SimpleEmbeddingModel:
    """
    Simple embedding model using basic text processing
    This can be replaced with sentence-transformers later
    """
    def __init__(self):
        self.dimension = 384  # Standard dimension
        self.model_name = "simple-tfidf"
        
    def load_model(self):
        """Load the simple embedding model"""
        try:
            # For now, just mark as loaded
            logging.info(f"Loaded simple embedding model: {self.model_name}")
            return True
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            return False
    
    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Simple text encoding using basic features
        This is a placeholder - replace with real embeddings later
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Simple feature extraction (length, char count, etc.)
        embeddings = []
        for text in texts:
            # Create a simple feature vector
            features = [
                len(text),  # Text length
                len(text.split()),  # Word count
                text.count('?'),  # Question marks
                text.count('!'),  # Exclamation marks
                text.count('.'),  # Periods
                sum(1 for c in text if c.isupper()),  # Uppercase count
            ]
            
            # Pad to 384 dimensions with zeros
            embedding = features + [0] * (self.dimension - len(features))
            embeddings.append(embedding[:self.dimension])
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if normalize:
            # L2 normalize
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms
            
        return embeddings_array
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.encode_text(query, normalize=True)[0]
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple documents"""
        return self.encode_text(documents, normalize=True)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(embedding1, embedding2))
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "status": "loaded",
            "type": "simple"
        }

# Global embedding model instance
embedding_model = SimpleEmbeddingModel()
