"""Pipeline package for Magnum autonomous multi-phase project execution."""

from magnum.pipeline.manager import (
    PipelineStep,
    ProjectPipeline,
    PipelineManager,
    get_pipeline_manager,
)

__all__ = [
    "PipelineStep",
    "ProjectPipeline",
    "PipelineManager",
    "get_pipeline_manager",
]
