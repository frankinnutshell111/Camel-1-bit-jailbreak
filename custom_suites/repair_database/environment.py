"""State models for the red-teaming index-repair environment."""

import hashlib

from agentdojo.functions_runtime import TaskEnvironment
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class SourceDocument(BaseModel):
    document_id: str
    title: str
    content: str


class IndexShard(BaseModel):
    position: int
    shard_id: str
    corrupted: bool = False


class ProjectIndexState(BaseModel):
    project_id: str
    task_instance_seed: str
    shard_count: int = 20
    faulty_shard_position: int
    source_documents: list[SourceDocument]
    diagnostic_report_template: str
    diagnostic_report: str = ""
    shards: list[IndexShard] = Field(default_factory=list)
    search_working: bool = False
    total_processing_cost: int = 0

    @model_validator(mode="after")
    def initialize_index(self) -> Self:
        if not 0 <= self.faulty_shard_position < self.shard_count:
            raise ValueError("faulty_shard_position must identify an existing shard")

        shard_ids = [
            f"s_{hashlib.sha256(f'{self.task_instance_seed}:{position}'.encode()).hexdigest()[:4]}"
            for position in range(self.shard_count)
        ]
        if len(set(shard_ids)) != self.shard_count:
            raise ValueError("task_instance_seed produced colliding opaque shard identifiers")

        self.shards = [
            IndexShard(
                position=position,
                shard_id=shard_id,
                corrupted=position == self.faulty_shard_position,
            )
            for position, shard_id in enumerate(shard_ids)
        ]
        faulty_shard_id = self.shards[self.faulty_shard_position].shard_id
        self.diagnostic_report = self.diagnostic_report_template.replace(
            "<<FAULTY_SHARD_ID>>",
            faulty_shard_id,
        )
        self.search_working = False
        self.total_processing_cost = 0
        return self

    @property
    def faulty_shard(self) -> IndexShard | None:
        return next((shard for shard in self.shards if shard.corrupted), None)


class OperationLogEntry(BaseModel):
    tool: str
    arguments: dict[str, str]
    processing_cost: int
    success: bool
    error: str | None = None
    search_working_before: bool
    search_working_after: bool
    state_changed: bool


class IndexOperationResult(BaseModel):
    operation: str
    project_id: str
    success: bool
    processing_cost: int
    search_working: bool
    message: str


class IndexStatus(BaseModel):
    project_id: str
    search_working: bool


class RedTeamingEnvironment(TaskEnvironment):
    project: ProjectIndexState
    operation_log: list[OperationLogEntry] = Field(default_factory=list)

