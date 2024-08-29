from pydantic import BaseModel, Field
from typing import Any, Dict, Union, List


class Document(BaseModel):
    document_title: str = Field(default="")
    document_text: str = Field(default="")


class ModelName(BaseModel):
    model_name: str = Field(default="meta-llama/llama-3-8b-instruct")


class PromptTemplate(BaseModel):
    prompt_template: str = Field(default="")


class SummaryEvaluationRequest(BaseModel):
    summary_model_name_list: List[str]
    summary_prompt_template_list: List[str]
    evaluate_summary_model_name_list: List[str]
    evaluate_summary_sentence_prompt_template_list: List[str]
    documents: Union[list[Document], None] = None


class ModelMetrics(BaseModel):
    summary_model_name: str
    summary_prompt_template: str
    document_title: str
    summary: str
    evaluate_summary_model_name: str
    evaluate_summary_sentence_prompt_template: str
    correct_sentences: int
    wrong_sentences: int
    summary_sentences_statistics: List[Dict]


class SummaryEvaluationResponse(BaseModel):
    model_metrics_list: list[ModelMetrics]
