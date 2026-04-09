"""
Точка входа для Uvicorn: `uvicorn main:app --reload` из корня репозитория.

Экспортирует FastAPI-приложение из пакета.
"""

from hallucination_detector.api.app import app

__all__ = ["app"]