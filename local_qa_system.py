#!/usr/bin/env python3

"""
Local Q&A System using trained vector database
Works without external dependencies
"""

from local_trainer import LocalVectorDB
from models.simple_embedding import embedding_model
from openai import OpenAI
import os
import json
from typing import List
from dotenv import load_dotenv

class LocalQASystem:
    def __init__(self):
        load_dotenv()
        self.local_db = LocalVectorDB()
        self.embedding_model = embedding_model
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
        )
        
    def initialize(self):
        """Initialize the local Q&A system"""
        print("🔄 Initializing Local Q&A System...")
        
        # Load embedding model
        if not self.embedding_model.load_model():
            print("❌ Failed to load embedding model")
            return False
        
        # Load vector database
        if not self.local_db.load("local_vector_db.json"):
            print("❌ Failed to load local vector database")
            print("💡 Run local_trainer.py first to create the database")
            return False
        
        stats = self.local_db.get_stats()
        print(f"✅ Local Q&A System ready! Database stats: {stats}")
        return True
    
    def search_knowledge_base(self, question: str, top_k: int = 3):
        """Search the local knowledge base for relevant Q&A pairs"""
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode_query(question)
            
            # Search in local database
            results = self.local_db.search_similar(question_embedding.tolist(), top_k=top_k)
            
            return results
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return []
    
    def answer_question(self, question: str, use_rag: bool = True):
        """Answer a question using the local knowledge base"""
        try:
            print(f"\\n🔍 Processing: '{question}'")
            
            if use_rag:
                # RAG approach: Retrieve + Generate
                print("🔍 Searching local knowledge base...")
                
                results = self.search_knowledge_base(question, top_k=3)
                
                if results:
                    print(f"📚 Found {len(results)} relevant entries")
                    
                    # Show top results
                    print("\\n📋 Most relevant Q&A pairs:")
                    for i, result in enumerate(results[:2], 1):
                        metadata = result['metadata']
                        print(f"  {i}. Score: {result['score']:.3f}")
                        print(f"     Q: {metadata['question'][:100]}...")
                        print(f"     A: {metadata['answer'][:100]}...")
                    
                    # Prepare context for LLM
                    context = "\\n\\n".join([
                        f"Q: {r['metadata']['question']}\\nA: {r['metadata']['answer']}"
                        for r in results[:2]
                    ])
                    
                    # Generate answer with context
                    system_prompt = f"""You are a helpful AI assistant. Use the following Q&A examples from the knowledge base to answer the user's question. 

Knowledge Base Context:
{context}

Instructions:
- Answer based on the knowledge base when possible
- If the examples are relevant, use them to inform your response
- If no direct match, provide a helpful general response
- Be concise and accurate"""

                    response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        temperature=0.7,
                        model="openai/gpt-4.1"
                    )
                    
                    answer = response.choices[0].message.content
                    print(f"\\n🤖 Answer: {answer}")
                    
                    return answer, results
                else:
                    print("📭 No relevant entries found in knowledge base")
                    return self.answer_question(question, use_rag=False)
            else:
                # Direct LLM approach
                print("🤖 Generating direct answer...")
                
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful AI assistant. Provide clear, accurate answers to questions."
                        },
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7,
                    model="openai/gpt-4.1"
                )
                
                answer = response.choices[0].message.content
                print(f"\\n🤖 Answer: {answer}")
                
                return answer, []
                
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"❌ Error: {error_msg}")
            return error_msg, []
    
    def interactive_mode(self):
        """Interactive Q&A mode"""
        print("\\n💬 Interactive Q&A Mode")
        print("Commands: 'quit' to exit, 'rag' to toggle RAG mode, 'stats' for database info")
        print("-" * 60)
        
        use_rag = True
        
        while True:
            mode_indicator = "RAG" if use_rag else "Direct"
            question = input(f"\\n🔹 You ({mode_indicator}): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\\n👋 Goodbye!")
                break
            elif question.lower() == 'rag':
                use_rag = not use_rag
                print(f"🔄 Switched to {'RAG' if use_rag else 'Direct'} mode")
                continue
            elif question.lower() == 'stats':
                stats = self.local_db.get_stats()
                print(f"📊 Database Stats: {stats}")
                continue
            elif not question:
                print("⚠️ Please enter a question!")
                continue
            
            self.answer_question(question, use_rag=use_rag)
    
    def batch_test(self, test_questions: List[str]):
        """Test multiple questions"""
        print(f"\\n🧪 Testing {len(test_questions)} questions:")
        print("-" * 50)
        
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"\\n{i}. Testing: {question}")
            answer, search_results = self.answer_question(question, use_rag=True)
            results.append({
                'question': question,
                'answer': answer,
                'search_results': search_results
            })
        
        return results

def main():
    qa_system = LocalQASystem()
    
    if not qa_system.initialize():
        print("\\n💡 To create the knowledge base, run:")
        print("   python local_trainer.py")
        return
    
    print("\\nChoose mode:")
    print("1. Interactive chat")
    print("2. Single question")
    print("3. Test with sample questions")
    print("4. Search knowledge base only")
    
    choice = input("\\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        qa_system.interactive_mode()
        
    elif choice == "2":
        question = input("\\n❓ Enter your question: ")
        qa_system.answer_question(question)
        
    elif choice == "3":
        test_questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is natural language processing?",
            "Tell me about AI applications"
        ]
        qa_system.batch_test(test_questions)
        
    elif choice == "4":
        question = input("\\n🔍 Search query: ")
        results = qa_system.search_knowledge_base(question)
        
        print(f"\\n📋 Search results for: '{question}'")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"\\n{i}. Score: {result['score']:.3f}")
            print(f"   Q: {metadata['question']}")
            print(f"   A: {metadata['answer']}")
            print(f"   Category: {metadata.get('category', 'N/A')}")
        
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
