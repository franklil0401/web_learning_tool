# -*- coding: utf-8 -*-
"""Student question handling module"""

from rag.pipeline import RAGPipeline


class StudentQuestionHandler:
    def __init__(self):
        self.pipeline = RAGPipeline()
    
    def handle_question(self, question: str) -> str:
        """Handle student question and generate answer"""
        if not question or not question.strip():
            return "Please enter a valid question"
        
        try:
            # Build LLM context
            context = self.pipeline.build_llm_context(question, retrieve_top_k=10, rerank_top_k=3)
            
            if not context:
                return "Sorry, I don't have enough course content to answer this question. Please let the teacher explain the relevant content first."
            
            # Generate answer using Qwen model
            answer = self.pipeline.answer_with_qwen(question, retrieve_top_k=10, rerank_top_k=3)
            return answer
        except Exception as e:
            print(f"Error processing question: {e}")
            return "Error processing question, please try again later"
    
    def get_relevant_context(self, question: str) -> str:
        """Get relevant course content for the question"""
        if not question or not question.strip():
            return ""
        
        try:
            context = self.pipeline.build_llm_context(question, retrieve_top_k=10, rerank_top_k=3)
            return context
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""
    
    def clear_history(self) -> None:
        """Clear all course history information"""
        try:
            self.pipeline.clear_history()
            print("Course history cleared successfully")
        except Exception as e:
            print(f"Error clearing history: {e}")


if __name__ == "__main__":
    handler = StudentQuestionHandler()
    
    print("AI Live Course Tutoring System - Student Question Interface")
    print("Enter 'exit' to quit the system")
    
    while True:
        question = input("Please enter your question: ")
        if question.lower() == 'exit':
            break
        
        answer = handler.handle_question(question)
        print("\nAI Answer:")
        print(answer)
        print("\n" + "-" * 50 + "\n")
