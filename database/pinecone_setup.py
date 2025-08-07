import pinecone
from config import Config
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

class PineconeManager:
    def __init__(self):
        self.api_key = Config.PINECONE_API_KEY
        self.environment = Config.PINECONE_ENVIRONMENT
        self.index_name = Config.PINECONE_INDEX_NAME
        self.dimension = Config.EMBEDDING_DIMENSION
        self.index = None
        
    def initialize(self):
        """Initialize Pinecone connection"""
        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists, create if not
            if self.index_name not in pinecone.list_indexes():
                self.create_index()
            
            self.index = pinecone.Index(self.index_name)
            logging.info(f"Pinecone initialized successfully with index: {self.index_name}")
            return True
        except Exception as e:
            logging.error(f"Error initializing Pinecone: {e}")
            return False
    
    def create_index(self):
        """Create Pinecone index"""
        try:
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                shards=1,
                replicas=1
            )
            logging.info(f"Created Pinecone index: {self.index_name}")
        except Exception as e:
            logging.error(f"Error creating Pinecone index: {e}")
            raise e
    
    def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        """
        Upsert vectors to Pinecone index
        
        Args:
            vectors: List of tuples (id, vector, metadata)
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            # Batch upsert for efficiency
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logging.info(f"Upserted {len(vectors)} vectors to Pinecone")
            return True
        except Exception as e:
            logging.error(f"Error upserting vectors: {e}")
            return False
    
    def search_similar(self, query_vector: List[float], top_k: int = 10, filter_dict: Dict = None):
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            search_kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False
            }
            
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            results = self.index.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            for match in results["matches"]:
                formatted_results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {})
                })
            
            return formatted_results
        except Exception as e:
            logging.error(f"Error searching vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by IDs"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            self.index.delete(ids=ids)
            logging.info(f"Deleted {len(ids)} vectors from Pinecone")
            return True
        except Exception as e:
            logging.error(f"Error deleting vectors: {e}")
            return False
    
    def get_index_stats(self):
        """Get index statistics"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logging.error(f"Error getting index stats: {e}")
            return None
    
    def reset_index(self):
        """Delete all vectors in the index"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            self.index.delete(delete_all=True)
            logging.info("Reset Pinecone index - deleted all vectors")
            return True
        except Exception as e:
            logging.error(f"Error resetting index: {e}")
            return False

# Initialize Pinecone manager
pinecone_manager = PineconeManager()
