import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import json
from pathlib import Path
from models.simple_embedding import embedding_model
from database.postgres_setup import db_manager
from database.pinecone_setup import pinecone_manager

class DataProcessor:
    def __init__(self):
        self.embedding_model = embedding_model
        self.db_manager = db_manager
        self.pinecone_manager = pinecone_manager
        
    def load_sample_data(self) -> List[Dict[str, Any]]:
        """
        Load sample documents for testing
        Replace this with your actual data loading logic
        """
        sample_documents = [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "metadata": {"category": "AI", "difficulty": "beginner"}
            },
            {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, particularly useful for image and text processing.",
                "metadata": {"category": "AI", "difficulty": "intermediate"}
            },
            {
                "title": "Natural Language Processing",
                "content": "NLP combines computational linguistics with machine learning to help computers understand, interpret, and generate human language in a valuable way.",
                "metadata": {"category": "NLP", "difficulty": "intermediate"}
            },
            {
                "title": "Computer Vision Applications",
                "content": "Computer vision enables machines to interpret and understand visual information from the world, including image recognition, object detection, and scene understanding.",
                "metadata": {"category": "CV", "difficulty": "advanced"}
            },
            {
                "title": "Data Science Process",
                "content": "Data science involves collecting, cleaning, analyzing, and interpreting large amounts of data to extract meaningful insights and support decision-making.",
                "metadata": {"category": "Data Science", "difficulty": "beginner"}
            }
        ]
        
        logging.info(f"Loaded {len(sample_documents)} sample documents")
        return sample_documents
    
    def load_data_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from a file (CSV, JSON, etc.)
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of document dictionaries
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                documents = df.to_dict('records')
                
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                    
            elif file_path.suffix.lower() == '.jsonl':
                documents = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        documents.append(json.loads(line.strip()))
                        
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logging.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess documents before indexing
        
        Args:
            documents: List of raw documents
            
        Returns:
            List of preprocessed documents
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                # Ensure required fields exist
                if 'content' not in doc:
                    logging.warning(f"Document {i} missing 'content' field, skipping")
                    continue
                
                # Clean and prepare text
                content = str(doc['content']).strip()
                title = str(doc.get('title', f'Document {i}')).strip()
                
                # Combine title and content for better search
                combined_text = f"{title}. {content}"
                
                processed_doc = {
                    'id': doc.get('id', f'doc_{i}'),
                    'title': title,
                    'content': content,
                    'combined_text': combined_text,
                    'metadata': doc.get('metadata', {}),
                    'original_index': i
                }
                
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logging.error(f"Error preprocessing document {i}: {e}")
                continue
        
        logging.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs
    
    def generate_embeddings(self, documents: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Generate embeddings for documents
        
        Args:
            documents: List of preprocessed documents
            
        Returns:
            Tuple of (embeddings_list, documents_with_ids)
        """
        try:
            # Load embedding model
            if not self.embedding_model.load_model():
                raise Exception("Failed to load embedding model")
            
            # Extract texts for embedding
            texts = [doc['combined_text'] for doc in documents]
            
            # Generate embeddings
            logging.info("Generating embeddings...")
            embeddings = self.embedding_model.encode_documents(texts)
            
            # Convert to list of arrays
            embeddings_list = [emb for emb in embeddings]
            
            logging.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list, documents
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return [], []
    
    def store_in_database(self, documents: List[Dict[str, Any]]) -> List[int]:
        """
        Store documents in PostgreSQL database
        
        Args:
            documents: List of documents to store
            
        Returns:
            List of database IDs
        """
        try:
            # Create tables if they don't exist
            self.db_manager.create_tables()
            
            doc_ids = []
            for doc in documents:
                doc_id = self.db_manager.insert_document(
                    title=doc['title'],
                    content=doc['content'],
                    metadata=doc['metadata']
                )
                if doc_id:
                    doc_ids.append(doc_id)
            
            logging.info(f"Stored {len(doc_ids)} documents in database")
            return doc_ids
            
        except Exception as e:
            logging.error(f"Error storing documents in database: {e}")
            return []
    
    def store_in_pinecone(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray], doc_ids: List[int]):
        """
        Store embeddings in Pinecone vector database
        
        Args:
            documents: List of documents
            embeddings: List of embeddings
            doc_ids: List of database IDs
        """
        try:
            # Initialize Pinecone
            if not self.pinecone_manager.initialize():
                raise Exception("Failed to initialize Pinecone")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, doc_ids)):
                vector_data = (
                    str(doc_id),  # Use database ID as vector ID
                    embedding.tolist(),  # Convert numpy array to list
                    {
                        'title': doc['title'],
                        'doc_id': doc_id,
                        'category': doc['metadata'].get('category', ''),
                        'difficulty': doc['metadata'].get('difficulty', '')
                    }
                )
                vectors.append(vector_data)
            
            # Upsert vectors
            success = self.pinecone_manager.upsert_vectors(vectors)
            if success:
                logging.info(f"Stored {len(vectors)} vectors in Pinecone")
            else:
                logging.error("Failed to store vectors in Pinecone")
                
        except Exception as e:
            logging.error(f"Error storing vectors in Pinecone: {e}")
    
    def process_and_index_data(self, data_source: str = None) -> bool:
        """
        Complete pipeline to process and index data
        
        Args:
            data_source: Path to data file, or None for sample data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logging.info("Starting data processing pipeline...")
            
            # Load data
            if data_source:
                documents = self.load_data_from_file(data_source)
            else:
                documents = self.load_sample_data()
            
            if not documents:
                logging.error("No documents to process")
                return False
            
            # Preprocess documents
            processed_docs = self.preprocess_documents(documents)
            if not processed_docs:
                logging.error("No documents after preprocessing")
                return False
            
            # Generate embeddings
            embeddings, _ = self.generate_embeddings(processed_docs)
            if not embeddings:
                logging.error("Failed to generate embeddings")
                return False
            
            # Store in database
            doc_ids = self.store_in_database(processed_docs)
            if not doc_ids:
                logging.error("Failed to store documents in database")
                return False
            
            # Store in Pinecone
            self.store_in_pinecone(processed_docs, embeddings, doc_ids)
            
            logging.info("✅ Data processing pipeline completed successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error in data processing pipeline: {e}")
            return False

# Global data processor instance
data_processor = DataProcessor()
