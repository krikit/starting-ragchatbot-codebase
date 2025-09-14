"""
Test course data loading to identify why 'query failed' occurs.
This test examines the course loading process in detail.
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch
from rag_system import RAGSystem
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore


class TestDataLoading:
    """Test course data loading process"""

    def setup_method(self):
        """Setup for data loading tests"""
        self.config = Config()
        self.config.ANTHROPIC_API_KEY = "test-key"

    def test_docs_directory_content(self):
        """Examine what's in the docs directory"""
        docs_path = os.path.join(os.path.dirname(__file__), '../../docs')

        if not os.path.exists(docs_path):
            pytest.skip("No docs directory found")

        files = os.listdir(docs_path)
        course_files = [f for f in files if f.endswith(('.txt', '.pdf', '.docx'))]

        print(f"📁 Found {len(course_files)} course files in docs:")
        for file in course_files:
            file_path = os.path.join(docs_path, file)
            size = os.path.getsize(file_path)
            print(f"   - {file} ({size} bytes)")

            # Check if files are readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_sample = f.read(200)  # First 200 chars
                    print(f"     ✅ Readable: {content_sample[:50]}...")
            except Exception as e:
                print(f"     ❌ Read error: {e}")

        assert len(course_files) > 0, "No course files found to test with"

    def test_document_processor_on_real_files(self):
        """Test document processor with actual course files"""
        docs_path = os.path.join(os.path.dirname(__file__), '../../docs')

        if not os.path.exists(docs_path):
            pytest.skip("No docs directory found")

        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]

        if not course_files:
            pytest.skip("No .txt course files found")

        # Test processing first file
        test_file = os.path.join(docs_path, course_files[0])
        print(f"🔍 Testing document processing on: {course_files[0]}")

        try:
            course, chunks = processor.process_course_document(test_file)

            if course:
                print(f"   ✅ Course parsed: {course.title}")
                print(f"   ✅ Instructor: {course.instructor}")
                print(f"   ✅ Lessons: {len(course.lessons)}")
                print(f"   ✅ Chunks created: {len(chunks)}")

                # Show first chunk sample
                if chunks:
                    print(f"   📝 First chunk sample: {chunks[0].content[:100]}...")

                return course, chunks
            else:
                print(f"   ❌ Failed to parse course from {course_files[0]}")
                pytest.fail("Document processor failed to parse course")

        except Exception as e:
            print(f"   ❌ Document processing error: {e}")
            pytest.fail(f"Document processor raised exception: {e}")

    def test_vector_store_data_loading(self):
        """Test loading course data into vector store"""
        docs_path = os.path.join(os.path.dirname(__file__), '../../docs')

        if not os.path.exists(docs_path):
            pytest.skip("No docs directory found")

        # Create fresh vector store
        vector_store = VectorStore("./test_data_loading", "all-MiniLM-L6-v2", 5)
        vector_store.clear_all_data()  # Start fresh

        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]

        if not course_files:
            pytest.skip("No .txt course files found")

        print(f"📚 Loading {len(course_files)} course files into vector store...")

        successful_loads = 0
        total_chunks = 0

        for file_name in course_files:
            file_path = os.path.join(docs_path, file_name)
            print(f"   Processing: {file_name}")

            try:
                course, chunks = processor.process_course_document(file_path)

                if course and chunks:
                    # Add to vector store
                    vector_store.add_course_metadata(course)
                    vector_store.add_course_content(chunks)

                    successful_loads += 1
                    total_chunks += len(chunks)
                    print(f"     ✅ Added course '{course.title}' with {len(chunks)} chunks")
                else:
                    print(f"     ❌ Failed to process {file_name}")

            except Exception as e:
                print(f"     ❌ Error processing {file_name}: {e}")

        print(f"\n📊 Loading summary:")
        print(f"   - Files processed: {len(course_files)}")
        print(f"   - Successful loads: {successful_loads}")
        print(f"   - Total chunks: {total_chunks}")
        print(f"   - Final course count: {vector_store.get_course_count()}")

        # Verify data was loaded
        assert vector_store.get_course_count() > 0, "No courses were loaded into vector store"

        # Test search with loaded data
        results = vector_store.search("Python")
        print(f"   - Search test results: {len(results.documents)} documents found")

        return vector_store

    def test_rag_system_auto_loading(self):
        """Test if RAG system automatically loads course data"""
        with patch('rag_system.AIGenerator') as mock_ai:
            mock_ai.return_value = Mock()

            # Initialize RAG system
            rag_system = RAGSystem(self.config)

            initial_count = rag_system.vector_store.get_course_count()
            print(f"📈 Initial course count after RAG system init: {initial_count}")

            # Try manual course loading
            docs_path = os.path.join(os.path.dirname(__file__), '../../docs')
            if os.path.exists(docs_path):
                print(f"🔄 Manually loading courses from {docs_path}")
                added_courses, added_chunks = rag_system.add_course_folder(docs_path)
                final_count = rag_system.vector_store.get_course_count()

                print(f"   - Courses added: {added_courses}")
                print(f"   - Chunks added: {added_chunks}")
                print(f"   - Final course count: {final_count}")

                if added_courses == 0 and final_count == 0:
                    print("❌ RAG system failed to load any course data")
                else:
                    print("✅ RAG system successfully loaded course data")

                return final_count > 0
            else:
                print("❌ No docs directory found for RAG system loading")
                return False

    def test_search_tool_with_loaded_data(self):
        """Test search tool behavior with actual loaded data"""
        # First load data
        try:
            vector_store = self.test_vector_store_data_loading()
        except pytest.skip.Exception:
            pytest.skip("Cannot load data for search test")

        from search_tools import CourseSearchTool

        search_tool = CourseSearchTool(vector_store)

        # Test various search queries
        test_queries = [
            "Python",
            "programming",
            "introduction",
            "variables",
            "functions"
        ]

        print(f"🔍 Testing search with loaded data:")

        for query in test_queries:
            result = search_tool.execute(query)
            print(f"   Query: '{query}' -> {len(result)} chars returned")

            if "No relevant content found" in result:
                print(f"     ❌ No results for '{query}'")
            else:
                print(f"     ✅ Found results for '{query}'")

        # Test specific search scenarios
        specific_result = search_tool.execute("Python programming basics")
        print(f"\n📝 Detailed search result sample:")
        print(f"   Query: 'Python programming basics'")
        print(f"   Result length: {len(specific_result)} characters")
        print(f"   Result preview: {specific_result[:200]}...")

        assert specific_result != "", "Search should return some content with loaded data"
        assert "No relevant content found" not in specific_result, "Should find relevant content with loaded data"


if __name__ == "__main__":
    pytest.main([__file__ + "::TestDataLoading", "-v", "-s"])