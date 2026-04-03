#!/usr/bin/env python3
"""Index lore documents into ChromaDB for RAG retrieval.

Usage:
    python scripts/index_lore.py [--lore-dir path/to/lore]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import EmbeddingService
from src.rag.lore_indexer import LoreIndexer
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Index lore documents into ChromaDB")
    parser.add_argument("--lore-dir", type=Path, default=None, help="Path to lore documents")
    args = parser.parse_args()

    setup_logging()

    embedding_service = EmbeddingService()
    indexer = LoreIndexer(embedding_service=embedding_service)

    total = indexer.index_directory(lore_dir=args.lore_dir)
    print(f"\nIndexed {total} chunks from lore documents.")


if __name__ == "__main__":
    main()
