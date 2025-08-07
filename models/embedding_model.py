import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from config import Config

class EmbeddingModel:
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.model = None
        self.dimension = Config.EMBEDDING_DIMENSION
        
    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logging.info(f"Loaded embedding model: {self.model_name}")
            logging.info(f"Embedding dimension: {self.dimension}")
            return True
        except Exception as e:
            logging.error(f"Error loading embedding model: {e}")
            return False
    
    def encode_text(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into embeddings
        
        Args:
            texts: Single text string or list of texts
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise Exception("Model not loaded. Call load_model() first.")
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(
                texts, 
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 10
            )
            
            return embeddings
        except Exception as e:
            logging.error(f"Error encoding text: {e}")
            raise e
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query
        
        Args:
            query: Query string
            
        Returns:
            Query embedding as numpy array
        """
        return self.encode_text(query, normalize=True)[0]
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple documents in batches
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            
        Returns:
            Document embeddings as numpy array
        """
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.encode_text(batch, normalize=True)
            all_embeddings.append(batch_embeddings)
            
            if len(documents) > 100:
                logging.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        return np.vstack(all_embeddings)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(embedding1, embedding2))
    
    def get_model_info(self):
        """Get model information"""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "status": "loaded"
        }

# Global embedding model instance
embedding_model = EmbeddingModel()
