#!/usr/bin/env python3

"""
Local Vector Database Q&A Training System
Works without external vector database dependencies
"""

import json
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from models.simple_embedding import embedding_model

class LocalVectorDB:
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.dimension = 384
        
    def add_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        """Add vectors to local database"""
        for vector_id, vector, meta in vectors:
            self.vectors.append({
                'id': vector_id,
                'vector': np.array(vector),
                'metadata': meta
            })
            self.metadata.append(meta)
    
    def search_similar(self, query_vector: List[float], top_k: int = 5):
        """Search for similar vectors using cosine similarity"""
        if not self.vectors:
            return []
        
        query_vec = np.array(query_vector)
        similarities = []
        
        for item in self.vectors:
            # Calculate cosine similarity
            vec = item['vector']
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            
            similarities.append({
                'id': item['id'],
                'score': float(similarity),
                'metadata': item['metadata']
            })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        return similarities[:top_k]
    
    def save(self, filename: str):
        """Save database to file"""
        data = {
            'vectors': [{'id': v['id'], 'vector': v['vector'].tolist(), 'metadata': v['metadata']} 
                       for v in self.vectors],
            'dimension': self.dimension
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filename: str):
        """Load database from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.vectors = []
            for item in data['vectors']:
                self.vectors.append({
                    'id': item['id'],
                    'vector': np.array(item['vector']),
                    'metadata': item['metadata']
                })
            
            self.dimension = data['dimension']
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def get_stats(self):
        """Get database statistics"""
        return {
            'total_vectors': len(self.vectors),
            'dimension': self.dimension,
            'categories': list(set(v['metadata'].get('category', 'unknown') for v in self.vectors))
        }

def local_qa_training():
    print("🚀 Local Q&A Vector Database Training")
    print("=" * 60)
    
    # Initialize local vector database
    local_db = LocalVectorDB()
    
    # Get Q&A pairs
    print("\\n📝 Would you like to:")
    print("1. Enter your Q&A pairs manually")
    print("2. Use sample data for testing")
    print("3. Load existing qa_dataset.json")
    
    choice = input("\\nChoice (1-3): ").strip()
    
    qa_pairs = []
    
    if choice == "1":
        print("\\n📋 Enter your Question-Answer pairs (type 'done' when finished):")
        
        while True:
            print(f"\\n📝 Q&A Pair #{len(qa_pairs) + 1}:")
            question = input("❓ Question: ").strip()
            if question.lower() == 'done':
                break
                
            answer = input("💬 Answer: ").strip()
            if answer.lower() == 'done':
                break
                
            category = input("🏷️ Category (optional): ").strip() or "general"
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": category,
                "id": f"qa_{len(qa_pairs) + 1}"
            })
            
            print(f"✅ Added Q&A pair #{len(qa_pairs)}")
    
    elif choice == "3":
        # Load existing dataset
        try:
            with open("qa_dataset.json", 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            print(f"\\n📊 Loaded {len(qa_pairs)} Q&A pairs from qa_dataset.json")
        except FileNotFoundError:
            print("\\n❌ qa_dataset.json not found. Using sample data instead.")
            choice = "2"
    
    if choice == "2" or (choice == "3" and not qa_pairs):
        # Use sample data
        qa_pairs = [
            {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "category": "AI",
                "id": "qa_1"
            },
            {
                "question": "How does deep learning work?",
                "answer": "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. Each layer processes the data and passes it to the next layer.",
                "category": "AI",
                "id": "qa_2"
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language using computational linguistics and machine learning.",
                "category": "NLP",
                "id": "qa_3"
            }
        ]
        print(f"\\n📊 Using {len(qa_pairs)} sample Q&A pairs")
    
    if not qa_pairs:
        print("❌ No Q&A pairs to process!")
        return None
    
    # Load embedding model
    print("\\n🔄 Loading embedding model...")
    if not embedding_model.load_model():
        print("❌ Failed to load embedding model")
        return None
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    
    questions = [qa['question'] for qa in qa_pairs]
    answers = [qa['answer'] for qa in qa_pairs]
    combined_texts = [f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs]
    
    question_embeddings = embedding_model.encode_documents(questions)
    combined_embeddings = embedding_model.encode_documents(combined_texts)
    
    print(f"✅ Generated embeddings for {len(qa_pairs)} Q&A pairs")
    
    # Add to local vector database
    print("🔄 Building local vector database...")
    
    vectors = []
    for i, qa in enumerate(qa_pairs):
        # Question vector
        question_vector = (
            f"q_{qa['id']}",
            question_embeddings[i].tolist(),
            {
                "type": "question",
                "question": qa['question'],
                "answer": qa['answer'],
                "category": qa['category'],
                "qa_id": qa['id']
            }
        )
        vectors.append(question_vector)
        
        # Combined vector
        combined_vector = (
            f"c_{qa['id']}",
            combined_embeddings[i].tolist(),
            {
                "type": "combined",
                "question": qa['question'],
                "answer": qa['answer'],
                "category": qa['category'],
                "qa_id": qa['id']
            }
        )
        vectors.append(combined_vector)
    
    local_db.add_vectors(vectors)
    print(f"✅ Added {len(vectors)} vectors to local database")
    
    # Save databases
    print("💾 Saving databases...")
    local_db.save("local_vector_db.json")
    
    with open("qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print("✅ Databases saved!")
    
    # Test the system
    print("\\n🧪 Testing the trained system:")
    test_question = qa_pairs[0]['question']
    
    test_embedding = embedding_model.encode_query(test_question)
    results = local_db.search_similar(test_embedding.tolist(), top_k=3)
    
    print(f"\\n🔍 Test query: '{test_question}'")
    print("📋 Results:")
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"\\n{i}. Score: {result['score']:.3f}")
        print(f"   Q: {metadata['question']}")
        print(f"   A: {metadata['answer']}")
    
    # Show stats
    stats = local_db.get_stats()
    print(f"\\n📊 Database Stats: {stats}")
    
    print("\\n🎉 Local training completed successfully!")
    print(f"📊 Processed {len(qa_pairs)} Q&A pairs")
    print("🔍 Your local vector database is ready for queries!")
    
    return local_db

if __name__ == "__main__":
    local_qa_training()
