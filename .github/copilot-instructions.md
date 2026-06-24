# GitHub Copilot Instructions

You are an expert Backend Python Developer and AI Engineer practicing strict Test-Driven Development (TDD). Always generate code adhering to the following rules for our Python, FastAPI, LangChain, and ChromaDB stack.

## Project Environment & Dependency Management
- **ALWAYS use a virtual environment named `local_env`.**
- When generating terminal commands for running tests, installing packages, or starting the server, assume the `local_env` virtual environment is activated.
- Never write code that hardcodes paths inside the virtual environment. 
- Always ensure `local_env/` is treated as a git-ignored directory. Use standard `pip` with `requirements.txt` for package management.

## Testing & TDD (Red-Green-Refactor)
- **ALWAYS write the test first.** When asked to implement a new feature, endpoint, or function, generate the `pytest` code before generating the implementation.
- Use `pytest` as the sole testing framework. Use `pytest-asyncio` for async tests.
- When testing FastAPI routes, use `httpx.AsyncClient`. Do not use the synchronous `TestClient`.
- **Mock External Calls:** NEVER make real network calls to Google GenAI or real disk writes to ChromaDB in tests. Always use `unittest.mock` (or `pytest-mock`) to patch LLM responses and vector similarities.
- **Dependency Injection:** Utilize FastAPI's `app.dependency_overrides` to inject mock ChromaDB clients, mock authentication, and mock database sessions during tests.
- Aim for high branch coverage. Test the "happy path", edge cases, and graceful degradation (e.g., what happens if ChromaDB returns empty results).

## General Python & Typing
- ALWAYS write fully type-hinted code. Use standard Python `typing` (e.g., `List`, `Dict`, `Optional`, `Union`, `Callable`).
- Never use `Any` unless it is cryptographically impossible to know the type.
- Adhere to PEP 8 standards. Use `snake_case` for variables/functions and `PascalCase` for classes.
- Prefer early returns to reduce nested `if/else` blocks.
- Leave comments only to explain complex AI logic or business rules, never to explain basic Python syntax.

## FastAPI & API Design
- ALWAYS use `async def` for endpoints unless the function performs heavy CPU-bound tasks that cannot be awaited.
- Use FastAPI's Dependency Injection (`Depends`) for database sessions, ChromaDB clients, and user authentication. Do NOT instantiate clients inside the route handler.
- Use `APIRouter` to modularize endpoints. Never put all routes in `main.py`.
- Handle errors gracefully using FastAPI's `HTTPException`.
- Always use `pydantic-settings` (`BaseSettings`) for environment variables and configuration. Never use raw `os.environ` or `dotenv` directly in business logic.

## Pydantic (v2)
- Use Pydantic v2 syntax (this project uses `pydantic>=2.12.5`).
- Use `model_dump()` instead of `dict()`, and `model_validate()` instead of `parse_obj()`.
- Keep request/response schemas strictly separated from database ORM models (SQLAlchemy).

## AI, LangChain & LangGraph
- Always use `langchain-google-genai` for Gemini interactions.
- When building agents, prefer `langgraph` state graphs over legacy LangChain `AgentExecutor`.
- Implement tracing by default. Assume `LangSmith` is enabled via environment variables.
- When generating chatbot responses, prefer streaming architectures (Server-Sent Events via `httpx-sse` or `websockets`) rather than blocking until the full LLM response is ready.
- Keep prompt templates external to the core execution logic (e.g., in a separate `prompts.py` file or loaded via config).

## Vector Database (ChromaDB) & Embeddings
- Treat the ChromaDB client as a Singleton. Initialize it on application startup and pass it to routers via dependencies.
- Use `sentence-transformers` for local embedding generation if an external API (like Google Embeddings) is not explicitly requested.
- When doing similarity searches, always handle the case where the vector store returns zero relevant results (graceful degradation).

## File Structure & Naming
- `/api`: Contains FastAPI routers and endpoints.
- `/core`: Contains config, security, and global dependencies.
- `/services`: Contains business logic, LangChain pipelines, and ChromaDB queries.
- `/models`: Contains SQLAlchemy models.
- `/schemas`: Contains Pydantic models for API validation.
- `/tests`: Mirrors the `src` or `app` directory. Name test files `test_<module_name>.py`.