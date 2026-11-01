"""Expose the shared graph retriever at the original runtime import path."""

from app.rag.graph_retriever import GraphRetriever

__all__ = ["GraphRetriever"]
