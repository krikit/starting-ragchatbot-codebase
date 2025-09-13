# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system - a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. The system uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Architecture

The application follows a modular Python backend architecture with a vanilla JavaScript frontend:

### Backend Components (`backend/`)
- **`app.py`** - FastAPI application entry point with API endpoints (`/api/query`, `/api/courses`)
- **`rag_system.py`** - Main orchestrator that coordinates all components
- **`vector_store.py`** - ChromaDB interface with embedding operations using Sentence Transformers
- **`document_processor.py`** - Text chunking and course document parsing
- **`ai_generator.py`** - Anthropic Claude API integration for response generation
- **`session_manager.py`** - User session and conversation history management
- **`search_tools.py`** - Tool-based search functionality for agent interactions
- **`models.py`** - Pydantic data models (Course, Lesson, CourseChunk)
- **`config.py`** - Configuration management with environment variables

### Frontend (`frontend/`)
- **`index.html`** - Single-page application with chat interface
- **`script.js`** - JavaScript for API communication and UI interactions
- **`style.css`** - Complete styling for the web interface

### Data Storage
- **`docs/`** - Course material text files (auto-loaded on startup)
- **`backend/chroma_db/`** - ChromaDB vector database storage (created automatically)

## Common Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name
```

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Uses Python 3.13+ and uv package manager

## Key Configuration

The system uses a centralized configuration in `backend/config.py`:
- **Chunk size**: 800 characters with 100 character overlap
- **Embedding model**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Claude model**: claude-sonnet-4-20250514
- **Max search results**: 5 documents per query
- **Session history**: 2 previous messages retained

## Development Notes

- FastAPI serves both API endpoints and static frontend files
- ChromaDB automatically initializes on first run
- Documents in `docs/` folder are auto-loaded on application startup
- CORS is enabled for development with wildcard origins
- No test suite or linting configuration currently present
- Frontend uses vanilla JavaScript (no framework dependencies)