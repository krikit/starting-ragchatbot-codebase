#!/usr/bin/env python3
"""
Diagnostic script to check API setup and identify configuration issues.
Run this to verify your RAG chatbot setup before starting the application.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from config import Config
from rag_system import RAGSystem


def check_api_key():
    """Check if Anthropic API key is properly configured"""
    print("🔑 Checking API Configuration...")

    config = Config()

    if not config.ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY is not set!")
        print("   Create a .env file in the project root with:")
        print("   ANTHROPIC_API_KEY=your_api_key_here")
        return False

    if config.ANTHROPIC_API_KEY.startswith("sk-ant-"):
        print(f"✅ API key found (starts with sk-ant-)")
        return True
    else:
        print(f"⚠️  API key found but format looks unusual: {config.ANTHROPIC_API_KEY[:10]}...")
        return True


def check_course_data():
    """Check if course data is loaded"""
    print("\n📚 Checking Course Data...")

    config = Config()
    config.ANTHROPIC_API_KEY = "test-key"  # Use dummy key for data check

    try:
        with patch('rag_system.AIGenerator') as mock_ai:
            mock_ai.return_value = Mock()
            rag_system = RAGSystem(config)

            course_count = rag_system.vector_store.get_course_count()
            print(f"📊 Courses loaded: {course_count}")

            if course_count == 0:
                print("⚠️  No courses loaded. Attempting to load from docs/...")
                docs_path = Path(__file__).parent.parent / "docs"
                if docs_path.exists():
                    added_courses, added_chunks = rag_system.add_course_folder(str(docs_path))
                    print(f"   Added {added_courses} courses with {added_chunks} chunks")
                    return added_courses > 0
                else:
                    print("❌ No docs folder found!")
                    return False

            return True

    except Exception as e:
        print(f"❌ Error checking course data: {e}")
        return False


def test_search_functionality():
    """Test search functionality with loaded data"""
    print("\n🔍 Testing Search Functionality...")

    config = Config()
    config.ANTHROPIC_API_KEY = "test-key"

    try:
        from unittest.mock import Mock, patch

        with patch('rag_system.AIGenerator') as mock_ai:
            mock_ai.return_value = Mock()
            rag_system = RAGSystem(config)

            # Test search tool directly
            result = rag_system.search_tool.execute("Python programming")

            if "No relevant content found" in result:
                print("❌ Search returns no results")
                return False
            else:
                print(f"✅ Search working - returned {len(result)} characters")
                print(f"   Sample: {result[:100]}...")
                return True

    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def test_ai_generator():
    """Test AI generator with real API (if key available)"""
    print("\n🤖 Testing AI Generator...")

    config = Config()

    if not config.ANTHROPIC_API_KEY:
        print("⏭️  Skipping AI test - no API key")
        return True

    try:
        from ai_generator import AIGenerator

        ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        # Simple test without tools
        response = ai_gen.generate_response("What is 2 + 2?")

        if response and "4" in response:
            print("✅ AI Generator working correctly")
            return True
        else:
            print(f"⚠️  AI Generator response unexpected: {response}")
            return False

    except Exception as e:
        print(f"❌ AI Generator test failed: {e}")
        print(f"   This is likely an API key or network issue")
        return False


def main():
    """Run all diagnostic checks"""
    print("🏥 RAG Chatbot System Diagnostics")
    print("=" * 50)

    checks = [
        ("API Key", check_api_key),
        ("Course Data", check_course_data),
        ("Search Function", test_search_functionality),
        ("AI Generator", test_ai_generator)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY:")

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Your RAG system should work correctly.")
    else:
        print("\n⚠️  Some issues found. Address the failed checks above.")

        if not results[0][1]:  # API key failed
            print("\n🔧 QUICK FIX: Create .env file with your Anthropic API key:")
            print("   echo 'ANTHROPIC_API_KEY=your_key_here' > .env")

    return all_passed


if __name__ == "__main__":
    from unittest.mock import Mock, patch
    main()