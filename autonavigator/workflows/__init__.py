"""
Task workflows for Auto-Navigator.
"""

from .linkedin_workflow import LinkedInWorkflow
from .github_workflow import GitHubWorkflow
from .generic_workflow import GenericWorkflow

__all__ = ["LinkedInWorkflow", "GitHubWorkflow", "GenericWorkflow"]
