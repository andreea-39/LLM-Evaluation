from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field


class ExtractedPoses(BaseModel):
    Person_A: str = Field(default="")
    Person_B: str = Field(default="")


class ExtractedPosesEvaluationElement(BaseModel):
    dialogue: str = Field(default="")
    extracted_poses: ExtractedPoses


class ExtractedPosesEvaluationRequest(BaseModel):
    model_name_list: List[str]
    prompt_template_list: List[str]
    extracted_poses_evaluation_set: Union[
        list[ExtractedPosesEvaluationElement], None
    ] = None

    validation_type: str = Field(default="llm")  # alternative "regex"


class ModelMetrics(BaseModel):
    model_name: str = Field(default="")
    prompt_template: str = Field(default="")
    accuracy: Union[float, None] = Field(default=0)
    dialogues_analysis: List[Any]


class ExtractedPosesEvaluationResponse(BaseModel):
    model_metrics_list: list[ModelMetrics]


class ModelName(BaseModel):
    model_name: str = Field(default="meta-llama/llama-3-8b-instruct")


class PromptTemplate(BaseModel):
    prompt_template: str = Field(
        default="""
                Based on the following dialogue between Person_A and Person_B: $dialogue
                I want you to create a dictionary with
                the following format:
                    {
                    "Person_A": str,
                        "Person_B": str
                    },
                where each string represents a pose assigned from the list ["sitting", "standing", "kneeling", "laying down", "None"]. The pose is assigned by analyzing the content of the dialogue
                """
    )
