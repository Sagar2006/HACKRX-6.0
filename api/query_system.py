from typing import List, Dict, Any
import logging
from models.simple_embedding import embedding_model
from database.postgres_setup import db_manager
from database.pinecone_setup import pinecone_manager
from openai import OpenAI
from config import Config
import os

class QuerySystem:
    def __init__(self):
        self.embedding_model = embedding_model
        self.db_manager = db_manager
        self.pinecone_manager = pinecone_manager
        
        # Initialize GitHub Models API client
        self.client = OpenAI(
            base_url=Config.GITHUB_ENDPOINT,
            api_key=Config.GITHUB_TOKEN,
        )
        
    def initialize_system(self) -> bool:
        """Initialize all components of the query system"""
        try:
            # Load embedding model
            if not self.embedding_model.load_model():
                logging.error("Failed to load embedding model")
                return False
            
            # Initialize Pinecone
            if not self.pinecone_manager.initialize():
                logging.error("Failed to initialize Pinecone")
                return False
            
            logging.info("✅ Query system initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing query system: {e}")
            return False
    
    def search_similar_documents(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_query(query)
            
            # Search in Pinecone
            search_results = self.pinecone_manager.search_similar(
                query_vector=query_embedding.tolist(),
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # Get document details from database
            similar_docs = []
            for result in search_results:
                doc_id = int(result['metadata']['doc_id'])
                session = self.db_manager.get_session()
                
                try:
                    from database.postgres_setup import Document
                    document = session.query(Document).filter(Document.id == doc_id).first()
                    
                    if document:
                        similar_docs.append({
                            'id': document.id,
                            'title': document.title,
                            'content': document.content,
                            'metadata': document.doc_metadata,  # Changed from metadata to doc_metadata
                            'similarity_score': result['score'],
                            'relevance_rank': len(similar_docs) + 1
                        })
                finally:
                    session.close()
            
            logging.info(f"Found {len(similar_docs)} similar documents for query: {query}")
            return similar_docs
            
        except Exception as e:
            logging.error(f"Error searching similar documents: {e}")
            return []
    
    def generate_answer(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved documents for context
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context from retrieved documents
            context_text = "\\n\\n".join([
                f"Document {i+1}: {doc['title']}\\n{doc['content']}"
                for i, doc in enumerate(context_documents[:3])  # Use top 3 documents
            ])
            
            # Create system prompt
            system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided context documents. 
            
Use the following documents to answer the user's question:

{context_text}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Cite which document(s) you're referencing when possible"""

            # Generate response using GitHub Models
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                model=Config.GITHUB_MODEL
            )
            
            answer = response.choices[0].message.content
            logging.info(f"Generated answer for query: {query}")
            
            return answer
            
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while generating an answer: {str(e)}"
    
    def query_with_retrieval(self, query: str, top_k: int = 5, generate_answer: bool = True) -> Dict[str, Any]:
        """
        Complete query pipeline with retrieval and answer generation
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            generate_answer: Whether to generate an answer using LLM
            
        Returns:
            Dictionary with query results and answer
        """
        try:
            # Log the query
            query_id = self.db_manager.log_query(query)
            
            # Search for similar documents
            similar_docs = self.search_similar_documents(query, top_k)
            
            # Log query results
            if query_id and similar_docs:
                results_for_logging = [(doc['id'], doc['similarity_score']) for doc in similar_docs]
                self.db_manager.log_query_results(query_id, results_for_logging)
            
            # Generate answer if requested
            answer = None
            if generate_answer and similar_docs:
                answer = self.generate_answer(query, similar_docs)
            
            result = {
                'query': query,
                'query_id': query_id,
                'retrieved_documents': similar_docs,
                'answer': answer,
                'num_results': len(similar_docs)
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error in query pipeline: {e}")
            return {
                'query': query,
                'error': str(e),
                'retrieved_documents': [],
                'answer': None,
                'num_results': 0
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get Pinecone stats
            pinecone_stats = self.pinecone_manager.get_index_stats()
            
            # Get database stats
            session = self.db_manager.get_session()
            try:
                from database.postgres_setup import Document, Query
                doc_count = session.query(Document).count()
                query_count = session.query(Query).count()
            finally:
                session.close()
            
            return {
                'total_documents': doc_count,
                'total_queries': query_count,
                'pinecone_stats': pinecone_stats,
                'embedding_model': self.embedding_model.get_model_info()
            }
            
        except Exception as e:
            logging.error(f"Error getting system stats: {e}")
            return {'error': str(e)}

# Global query system instance
query_system = QuerySystem()
