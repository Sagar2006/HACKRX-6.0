import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from database.pinecone_setup import pinecone_manager
from config import Config

class QATrainingSystem:
    def __init__(self):
        self.pinecone_manager = pinecone_manager
        self.embedding_model = None
        self.qa_pairs = []
        
    def load_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load a proper sentence transformer model for better embeddings"""
        try:
            print(f"🔄 Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to simple model
            from models.simple_embedding import embedding_model
            self.embedding_model = embedding_model
            self.embedding_model.load_model()
            print("📝 Using simple embedding model as fallback")
            return True
    
    def add_qa_pairs_from_input(self):
        """Interactive method to add Q&A pairs"""
        print("\\n📝 Add your Question-Answer pairs (type 'done' when finished):")
        print("-" * 60)
        
        while True:
            print(f"\\n📋 Question-Answer Pair #{len(self.qa_pairs) + 1}:")
            
            question = input("❓ Question: ").strip()
            if question.lower() == 'done':
                break
            
            answer = input("💬 Answer: ").strip()
            if answer.lower() == 'done':
                break
            
            # Optional metadata
            category = input("🏷️ Category (optional): ").strip()
            difficulty = input("📊 Difficulty (beginner/intermediate/advanced, optional): ").strip()
            
            qa_pair = {
                "question": question,
                "answer": answer,
                "metadata": {
                    "category": category if category else "general",
                    "difficulty": difficulty if difficulty else "intermediate",
                    "id": f"qa_{len(self.qa_pairs) + 1}"
                }
            }
            
            self.qa_pairs.append(qa_pair)
            print(f"✅ Added Q&A pair #{len(self.qa_pairs)}")
        
        print(f"\\n🎯 Total Q&A pairs collected: {len(self.qa_pairs)}")
        return self.qa_pairs
    
    def add_qa_pairs_from_file(self, file_path: str):
        """Load Q&A pairs from a file"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.qa_pairs.extend(data)
                    else:
                        self.qa_pairs.append(data)
            
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    qa_pair = {
                        "question": row.get('question', ''),
                        "answer": row.get('answer', ''),
                        "metadata": {
                            "category": row.get('category', 'general'),
                            "difficulty": row.get('difficulty', 'intermediate'),
                            "id": f"qa_{len(self.qa_pairs) + 1}"
                        }
                    }
                    self.qa_pairs.append(qa_pair)
            
            print(f"✅ Loaded {len(self.qa_pairs)} Q&A pairs from {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False
    
    def add_qa_pairs_manually(self, qa_list: List[Dict[str, Any]]):
        """Add Q&A pairs from a provided list"""
        for i, qa in enumerate(qa_list):
            qa_pair = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "metadata": {
                    "category": qa.get("category", "general"),
                    "difficulty": qa.get("difficulty", "intermediate"),
                    "id": qa.get("id", f"qa_{len(self.qa_pairs) + 1}")
                }
            }
            self.qa_pairs.append(qa_pair)
        
        print(f"✅ Added {len(qa_list)} Q&A pairs manually")
        return True
    
    def generate_embeddings(self):
        """Generate embeddings for all Q&A pairs"""
        if not self.embedding_model:
            print("❌ Embedding model not loaded!")
            return False
        
        print("🔄 Generating embeddings for Q&A pairs...")
        
        # Prepare texts for embedding
        question_texts = []
        answer_texts = []
        combined_texts = []
        
        for qa in self.qa_pairs:
            question = qa['question']
            answer = qa['answer']
            combined = f"Question: {question} Answer: {answer}"
            
            question_texts.append(question)
            answer_texts.append(answer)
            combined_texts.append(combined)
        
        try:
            # Generate embeddings
            if hasattr(self.embedding_model, 'encode'):
                # Using sentence-transformers
                question_embeddings = self.embedding_model.encode(question_texts, show_progress_bar=True)
                answer_embeddings = self.embedding_model.encode(answer_texts, show_progress_bar=True)
                combined_embeddings = self.embedding_model.encode(combined_texts, show_progress_bar=True)
            else:
                # Using simple embedding model
                question_embeddings = self.embedding_model.encode_documents(question_texts)
                answer_embeddings = self.embedding_model.encode_documents(answer_texts)
                combined_embeddings = self.embedding_model.encode_documents(combined_texts)
            
            # Store embeddings in qa_pairs
            for i, qa in enumerate(self.qa_pairs):
                qa['question_embedding'] = question_embeddings[i].tolist()
                qa['answer_embedding'] = answer_embeddings[i].tolist()
                qa['combined_embedding'] = combined_embeddings[i].tolist()
            
            print(f"✅ Generated embeddings for {len(self.qa_pairs)} Q&A pairs")
            return True
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return False
    
    def create_pinecone_index(self, index_name: str = "qa-knowledge-base"):
        """Create and setup Pinecone index"""
        try:
            print(f"🔄 Setting up Pinecone index: {index_name}")
            
            # Update config with new index name
            Config.PINECONE_INDEX_NAME = index_name
            self.pinecone_manager.index_name = index_name
            
            # Initialize Pinecone
            if not self.pinecone_manager.initialize():
                print("❌ Failed to initialize Pinecone")
                return False
            
            print(f"✅ Pinecone index '{index_name}' ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Pinecone: {e}")
            return False
    
    def upload_to_pinecone(self):
        """Upload Q&A embeddings to Pinecone"""
        try:
            print("🔄 Uploading Q&A pairs to Pinecone...")
            
            vectors = []
            for i, qa in enumerate(self.qa_pairs):
                # Create vector for questions (for retrieval)
                question_vector = (
                    f"q_{qa['metadata']['id']}",
                    qa['question_embedding'],
                    {
                        "type": "question",
                        "question": qa['question'][:500],  # Truncate for metadata limits
                        "answer": qa['answer'][:500],
                        "category": qa['metadata']['category'],
                        "difficulty": qa['metadata']['difficulty'],
                        "qa_id": qa['metadata']['id']
                    }
                )
                vectors.append(question_vector)
                
                # Create vector for combined Q&A (for better context)
                combined_vector = (
                    f"c_{qa['metadata']['id']}",
                    qa['combined_embedding'],
                    {
                        "type": "combined",
                        "question": qa['question'][:400],
                        "answer": qa['answer'][:400],
                        "category": qa['metadata']['category'],
                        "difficulty": qa['metadata']['difficulty'],
                        "qa_id": qa['metadata']['id']
                    }
                )
                vectors.append(combined_vector)
            
            # Upload vectors to Pinecone
            success = self.pinecone_manager.upsert_vectors(vectors)
            
            if success:
                print(f"✅ Uploaded {len(vectors)} vectors to Pinecone")
                
                # Get index stats
                stats = self.pinecone_manager.get_index_stats()
                if stats:
                    print(f"📊 Index stats: {stats}")
                
                return True
            else:
                print("❌ Failed to upload vectors")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading to Pinecone: {e}")
            return False
    
    def save_qa_dataset(self, filename: str = "qa_dataset.json"):
        """Save Q&A pairs to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved Q&A dataset to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False
    
    def test_retrieval(self, test_question: str, top_k: int = 3):
        """Test the retrieval system with a question"""
        try:
            print(f"\\n🔍 Testing retrieval for: '{test_question}'")
            
            # Generate embedding for test question
            if hasattr(self.embedding_model, 'encode'):
                test_embedding = self.embedding_model.encode([test_question])[0]
            else:
                test_embedding = self.embedding_model.encode_query(test_question)
            
            # Search in Pinecone
            results = self.pinecone_manager.search_similar(
                query_vector=test_embedding.tolist(),
                top_k=top_k
            )
            
            print(f"📋 Top {len(results)} results:")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"\\n{i}. Score: {result['score']:.3f}")
                print(f"   Question: {metadata['question']}")
                print(f"   Answer: {metadata['answer']}")
                print(f"   Category: {metadata['category']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error testing retrieval: {e}")
            return []
    
    def complete_training_pipeline(self, method: str = "input"):
        """Complete training pipeline"""
        print("🚀 Starting Q&A Training Pipeline")
        print("=" * 60)
        
        # Step 1: Load embedding model
        if not self.load_embedding_model():
            return False
        
        # Step 2: Collect Q&A pairs
        if method == "input":
            self.add_qa_pairs_from_input()
        elif method == "file":
            file_path = input("📁 Enter file path: ")
            self.add_qa_pairs_from_file(file_path)
        
        if not self.qa_pairs:
            print("❌ No Q&A pairs collected!")
            return False
        
        # Step 3: Generate embeddings
        if not self.generate_embeddings():
            return False
        
        # Step 4: Setup Pinecone
        if not self.create_pinecone_index():
            return False
        
        # Step 5: Upload to Pinecone
        if not self.upload_to_pinecone():
            return False
        
        # Step 6: Save dataset
        self.save_qa_dataset()
        
        # Step 7: Test the system
        print("\\n🧪 Testing the trained system:")
        if self.qa_pairs:
            test_question = self.qa_pairs[0]['question']
            self.test_retrieval(test_question)
        
        print("\\n🎉 Training pipeline completed successfully!")
        print(f"📊 Total Q&A pairs processed: {len(self.qa_pairs)}")
        print("🔍 Your vector database is ready for queries!")
        
        return True

# Global training system instance
qa_trainer = QATrainingSystem()
