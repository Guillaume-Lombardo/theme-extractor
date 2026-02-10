"""Ingestion and text-cleaning pipeline for heterogeneous corpora."""

from theme_extractor.ingestion.pipeline import IngestionConfig, IngestionPipeline, run_ingestion

__all__ = ["IngestionConfig", "IngestionPipeline", "run_ingestion"]
