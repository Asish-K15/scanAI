"""
App service wrapper for Animalness Gate.
Provides convenient access consistent with other services in app.services.
"""

from animalness_gate.service import (
    AnimalnessGateConfig,
    AnimalnessGateService,
    get_animalness_gate_service,
)

__all__ = [
    "AnimalnessGateConfig",
    "AnimalnessGateService",
    "get_animalness_gate_service",
]
