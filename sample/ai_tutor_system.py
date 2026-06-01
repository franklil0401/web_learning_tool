# -*- coding: utf-8 -*-
"""AI live course tutoring system"""

import os
import threading
import time
from student_question_handler import StudentQuestionHandler


class AITutorSystem:
    def __init__(self):
        self.student_handler = StudentQuestionHandler()
        self.is_listening = False
        self.listener_thread = None
    
    def start_listening(self):
        """Start listening to live course content"""
        if self.is_listening:
            print("System is already listening...")
            return
        
        self.is_listening = True
        print("Starting to listen to live course content...")
        
        # Start voice listening thread
        self.listener_thread = threading.Thread(target=self._run_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
    
    def _run_listener(self):
        """Run voice listener"""
        try:
            # Import and run system audio listener
            import system_audio_listener_vosk
            system_audio_listener_vosk.main()
        except Exception as e:
            print(f"Error during listening: {e}")
            self.is_listening = False
    
    def stop_listening(self):
        """Stop listening to live course content"""
        if not self.is_listening:
            print("System is not listening...")
            return
        
        self.is_listening = False
        print("Stopping listening to live course content...")
        # Since system_audio_listener_vosk uses an infinite loop, wait for thread to end
        if self.listener_thread:
            self.listener_thread.join(timeout=5)
    
    def end_course(self):
        """End course and clear all history information"""
        print("Ending course and clearing history information...")
        # Stop listening if it's running
        if self.is_listening:
            self.stop_listening()
        # Clear history
        self.student_handler.clear_history()
        print("Course ended and history cleared successfully")
    
    def handle_student_question(self, question: str) -> str:
        """Handle student question"""
        return self.student_handler.handle_question(question)
    
    def run_interactive(self):
        """Run interactive system"""
        print("AI Live Course Tutoring System")
        print("=" * 50)
        print("1. Start listening to live course")
        print("2. Stop listening to live course")
        print("3. Student question")
        print("4. End course and clear history")
        print("5. Exit system")
        print("=" * 50)
        
        while True:
            choice = input("Please select an operation (1-5): ")
            
            if choice == "1":
                self.start_listening()
            elif choice == "2":
                self.stop_listening()
            elif choice == "3":
                question = input("Please enter your question: ")
                if question:
                    answer = self.handle_student_question(question)
                    print("\nAI Answer:")
                    print(answer)
                    print("\n" + "-" * 50 + "\n")
            elif choice == "4":
                self.end_course()
            elif choice == "5":
                print("Exiting system...")
                if self.is_listening:
                    self.stop_listening()
                break
            else:
                print("Invalid choice, please try again")


if __name__ == "__main__":
    # Set environment variables to enable live RAG
    os.environ["ENABLE_LIVE_RAG"] = "1"
    
    system = AITutorSystem()
    system.run_interactive()
