from __future__ import annotations

from dataclasses import dataclass

from .base import AgentContext
from .editorial import EditorialDecisionAgent, EditorialManager
from .publication import ProductionAgent, PublicationAgent
from .reviewers import REVIEWERS, ReviewerAgent


@dataclass
class AgentTeam:
    """The ten agents: 4 editorial/publication agents + the 6-reviewer pool."""

    editorial_manager: EditorialManager
    decision: EditorialDecisionAgent
    publication: PublicationAgent
    production: ProductionAgent
    reviewers: dict[str, ReviewerAgent]

    @classmethod
    def build(cls, ctx: AgentContext) -> "AgentTeam":
        return cls(
            editorial_manager=EditorialManager(ctx),
            decision=EditorialDecisionAgent(ctx),
            publication=PublicationAgent(ctx),
            production=ProductionAgent(ctx),
            reviewers={rid: ReviewerAgent(spec, ctx) for rid, spec in REVIEWERS.items()},
        )

    @property
    def agent_ids(self) -> list[str]:
        return [
            EditorialManager.agent_id,
            EditorialDecisionAgent.agent_id,
            PublicationAgent.agent_id,
            ProductionAgent.agent_id,
            *self.reviewers,
        ]


__all__ = ["AgentContext", "AgentTeam", "REVIEWERS"]
