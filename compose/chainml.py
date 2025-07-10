from typing import Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class DataType(str, Enum):
    """Supported data types for inputs and schema definitions."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


class ParserType(str, Enum):
    """Supported parser types for step outputs."""
    STRING = "string"
    JSON = "json"
    STRUCTURED = "structured"


class WorkflowInput(BaseModel):
    """Definition of a workflow input parameter."""
    type: DataType = Field(..., description="Data type of the input")
    description: str = Field(..., description="Description of the input's purpose")
    default: Optional[Union[str, int, float, bool]] = Field(None, description="Optional default value")


class WorkflowOutput(BaseModel):
    """Definition of a workflow output."""
    from_: str = Field(..., alias="from", description="Reference to step output (e.g., 'step_name.output' or 'step_name.output.field')")
    description: str = Field(..., description="Description of the output")


class StepCondition(BaseModel):
    """Conditional execution logic for a step."""
    when: str = Field(..., description="Jinja2 template expression that evaluates to boolean")


class StepParser(BaseModel):
    """Parser configuration for structured step outputs."""
    type: ParserType = Field(..., description="Type of parser to use")
    title: Optional[str] = Field(None, description="Optional name for the generated data model")
    schema: Optional[Union[str, dict[str, Any], list[str]]] = Field(None, description="Data structure definition for json/structured types")

    @validator('schema')
    def validate_schema_for_type(cls, v, values):
        """Ensure schema is provided for json and structured types."""
        parser_type = values.get('type')
        if parser_type in [ParserType.JSON, ParserType.STRUCTURED] and v is None:
            raise ValueError(f"Schema is required for parser type '{parser_type}'")
        return v


class WorkflowStep(BaseModel):
    """A single step in the workflow."""
    model: str = Field(..., description="LLM model to use for this step")
    description: str = Field(..., description="Description of what this step does")
    prompt: str = Field(..., description="Jinja2 template for the LLM prompt")
    depends_on: list[str] = Field(..., description="list of step names this step depends on")
    parser: Optional[StepParser] = Field(None, description="Optional parser configuration for structured outputs")
    condition: Optional[StepCondition] = Field(None, description="Optional conditional execution logic")

    @validator('depends_on')
    def validate_depends_on_is_list(cls, v):
        """Ensure depends_on is always a list, even if empty."""
        if not isinstance(v, list):
            raise ValueError("depends_on must be a list")
        return v


class Workflow(BaseModel):
    """Main workflow definition."""
    name: str = Field(..., description="Human-readable workflow name")
    description: str = Field(..., description="Description of what the workflow accomplishes")
    inputs: dict[str, WorkflowInput] = Field(..., description="Input parameters for the workflow")
    outputs: dict[str, WorkflowOutput] = Field(..., description="Output definitions for the workflow")
    steps: dict[str, WorkflowStep] = Field(..., description="Processing steps in the workflow")

    @validator('steps')
    def validate_dag_structure(cls, v):
        """Validate that the steps form a valid DAG (no cycles)."""
        # Build adjacency list
        graph = {step_name: step.depends_on for step_name, step in v.items()}
        
        # Check for self-dependencies
        for step_name, dependencies in graph.items():
            if step_name in dependencies:
                raise ValueError(f"Step '{step_name}' cannot depend on itself")
        
        # Check for cycles using DFS
        def has_cycle():
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {node: WHITE for node in graph}
            
            def dfs(node):
                if color[node] == GRAY:
                    return True  # Back edge found, cycle detected
                if color[node] == BLACK:
                    return False  # Already processed
                
                color[node] = GRAY
                for neighbor in graph.get(node, []):
                    if neighbor not in graph:
                        raise ValueError(f"Step '{node}' depends on undefined step '{neighbor}'")
                    if dfs(neighbor):
                        return True
                color[node] = BLACK
                return False
            
            for node in graph:
                if color[node] == WHITE:
                    if dfs(node):
                        return True
            return False
        
        if has_cycle():
            raise ValueError("Workflow contains circular dependencies")
        
        return v

    @validator('outputs')
    def validate_output_references(cls, v, values):
        """Validate that output references point to valid steps."""
        if 'steps' not in values:
            return v
        
        steps = values['steps']
        for output_name, output_def in v.items():
            ref = output_def.from_
            # Parse the reference (e.g., "step_name.output" or "step_name.output.field")
            parts = ref.split('.')
            if len(parts) < 2:
                raise ValueError(f"Output '{output_name}' reference '{ref}' must include step name and output")
            
            step_name = parts[0]
            if step_name not in steps:
                raise ValueError(f"Output '{output_name}' references undefined step '{step_name}'")
        
        return v


class ChainML(BaseModel):
    """Root ChainML document."""
    workflow: Workflow = Field(..., description="The workflow definition")

    class Config:
        """Pydantic configuration."""
        # Allow field aliases (like 'from' -> 'from_')
        allow_population_by_field_name = True
        # Validate assignment to catch issues early
        validate_assignment = True

    @classmethod
    def from_json(cls, json_str: str) -> "ChainML":
        """Create ChainML instance from JSON string."""
        import json
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChainML":
        """Create ChainML instance from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert ChainML instance to dictionary."""
        return self.dict(by_alias=True)

    def to_json(self, **kwargs) -> str:
        """Convert ChainML instance to JSON string."""
        return self.json(by_alias=True, **kwargs)

    def get_execution_order(self) -> list[str]:
        """Get topologically sorted order of steps for execution."""
        # Build adjacency list
        graph = {step_name: step.depends_on for step_name, step in self.workflow.steps.items()}
        
        # Topological sort using Kahn's algorithm
        in_degree = {node: 0 for node in graph}
        for dependencies in graph.values():
            for dep in dependencies:
                in_degree[dep] += 1
        
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result

    def get_parallel_groups(self) -> list[list[str]]:
        """Get groups of steps that can be executed in parallel."""
        execution_order = self.get_execution_order()
        groups = []
        current_group = []
        processed = set()
        
        for step_name in execution_order:
            step = self.workflow.steps[step_name]
            # Check if all dependencies are satisfied
            if all(dep in processed for dep in step.depends_on):
                current_group.append(step_name)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [step_name]
            
            processed.add(step_name)
        
        if current_group:
            groups.append(current_group)
        
        return groups

    def validate_step_references(self) -> bool:
        """Validate that all step references in prompts and conditions are valid."""
        import re
        
        all_steps = set(self.workflow.steps.keys())
        all_inputs = set(self.workflow.inputs.keys())
        
        # Pattern to find Jinja2 variable references
        var_pattern = r'\{\{\s*([^}]+)\s*\}\}'
        
        for step_name, step in self.workflow.steps.items():
            # Check prompt references
            prompt_vars = re.findall(var_pattern, step.prompt)
            for var in prompt_vars:
                # Parse variable reference (e.g., "inputs.url", "step_name.output")
                parts = var.split('.')[0].split('|')[0].strip()  # Remove filters and get base
                if parts.startswith('inputs'):
                    continue  # Input references are always valid if they exist
                elif parts not in all_steps and parts != step_name:
                    # Allow self-reference for some cases, but warn about others
                    if '.' in var and not var.startswith('inputs.'):
                        base_step = var.split('.')[0]
                        if base_step not in all_steps:
                            raise ValueError(f"Step '{step_name}' references undefined step '{base_step}' in prompt")
            
            # Check condition references if present
            if step.condition:
                condition_vars = re.findall(var_pattern, step.condition.when)
                for var in condition_vars:
                    parts = var.split('.')[0].split('|')[0].strip()
                    if parts.startswith('inputs'):
                        continue
                    elif parts not in all_steps and parts != step_name:
                        if '.' in var and not var.startswith('inputs.'):
                            base_step = var.split('.')[0]
                            if base_step not in all_steps:
                                raise ValueError(f"Step '{step_name}' references undefined step '{base_step}' in condition")
        
        return True
