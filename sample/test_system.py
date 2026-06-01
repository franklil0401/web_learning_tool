# -*- coding: utf-8 -*-
"""Test script for AI live course tutoring system"""

import os

print("Testing AI Live Course Tutoring System...")
print("=" * 50)

# Test 1: Check if required modules can be imported
try:
    from rag.pipeline import RAGPipeline
    print("? RAGPipeline imported successfully")
except Exception as e:
    print(f"? Failed to import RAGPipeline: {e}")

try:
    from student_question_handler import StudentQuestionHandler
    print("? StudentQuestionHandler imported successfully")
except Exception as e:
    print(f"? Failed to import StudentQuestionHandler: {e}")

# Test 2: Check if voice listening module is available
try:
    import system_audio_listener_vosk
    print("? system_audio_listener_vosk imported successfully")
except Exception as e:
    print(f"? Failed to import system_audio_listener_vosk: {e}")

print("=" * 50)
print("Test completed!")
