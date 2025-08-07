#!/usr/bin/env python3

"""
Smart Q&A System using trained vector database
"""

from training.qa_trainer import qa_trainer
from api.query_system import query_system
from openai import OpenAI
import os
from dotenv import load_dotenv

class SmartQASystem:
    def __init__(self):
        load_dotenv()
        self.qa_trainer = qa_trainer
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
        )
        
    def initialize(self):
        """Initialize the Q&A system"""
        print("🔄 Initializing Smart Q&A System...")
        
        # Load embedding model
        if not self.qa_trainer.load_embedding_model():
            return False
            
        # Initialize Pinecone
        if not self.qa_trainer.create_pinecone_index():
            return False
            
        print("✅ Smart Q&A System ready!")
        return True
    
    def answer_question(self, question: str, use_rag: bool = True):
        """Answer a question using the trained vector database"""
        try:
            if use_rag:
                # RAG approach: Retrieve + Generate
                print(f"🔍 Searching knowledge base for: '{question}'")
                
                # Search for relevant Q&A pairs
                results = self.qa_trainer.test_retrieval(question, top_k=3)
                
                if results:
                    # Prepare context from retrieved Q&A pairs
                    context = "\\n\\n".join([
                        f"Q: {r['metadata']['question']}\\nA: {r['metadata']['answer']}"
                        for r in results[:2]  # Use top 2 results
                    ])
                    
                    # Generate answer with context
                    system_prompt = f"""You are a helpful AI assistant. Use the following Q&A examples from the knowledge base to answer the user's question. If the examples don't directly answer the question, use them as context to provide a helpful response.

Knowledge Base Context:
{context}

Instructions:
- Answer based on the knowledge base when possible
- If no direct answer is available, provide general knowledge while noting the limitation
- Be concise and helpful"""

                    response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        temperature=0.7,
                        model="openai/gpt-4.1"
                    )
                    
                    answer = response.choices[0].message.content
                    
                    print(f"\\n🤖 Answer (RAG): {answer}")
                    print(f"\\n📚 Based on {len(results)} relevant entries from knowledge base")
                    
                else:
                    print("📭 No relevant entries found in knowledge base")
                    return self.answer_question(question, use_rag=False)
                    
            else:
                # Direct LLM approach
                print(f"🤖 Generating direct answer for: '{question}'")
                
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
                print(f"\\n🤖 Answer (Direct): {answer}")
            
            return answer
            
        except Exception as e:
            print(f"❌ Error answering question: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def interactive_chat(self):
        """Interactive chat mode"""
        print("\\n💬 Interactive Q&A Chat")
        print("Type 'quit' to exit, 'rag' to toggle RAG mode")
        print("-" * 50)
        
        use_rag = True
        
        while True:
            question = input(f"\\n🔹 You ({'RAG' if use_rag else 'Direct'}): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\\n👋 Goodbye!")
                break
            elif question.lower() == 'rag':
                use_rag = not use_rag
                print(f"🔄 Switched to {'RAG' if use_rag else 'Direct'} mode")
                continue
            elif not question:
                print("⚠️ Please enter a question!")
                continue
            
            self.answer_question(question, use_rag=use_rag)
    
    def batch_test(self, test_questions: list):
        """Test multiple questions"""
        print(f"\\n🧪 Testing {len(test_questions)} questions:")
        print("-" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\\n{i}. {question}")
            self.answer_question(question, use_rag=True)

def main():
    qa_system = SmartQASystem()
    
    if not qa_system.initialize():
        print("❌ Failed to initialize Q&A system")
        return
    
    print("\\nChoose mode:")
    print("1. Interactive chat")
    print("2. Single question")
    print("3. Test with sample questions")
    
    choice = input("\\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        qa_system.interactive_chat()
        
    elif choice == "2":
        question = input("\\n❓ Enter your question: ")
        qa_system.answer_question(question)
        
    elif choice == "3":
        test_questions = [
            "What is machine learning?",
            "How does deep learning work?",
            "What is the difference between AI and ML?",
            "How do you evaluate ML models?",
            "What are some applications of computer vision?"
        ]
        qa_system.batch_test(test_questions)
        
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
