from ..router import router
from evaluation_end_points.evaluate.evaluate_summary.schema import (
    SummaryEvaluationRequest,
    SummaryEvaluationResponse,
)
from evaluation_end_points.evaluate.evaluate_summary.service import wrapper


@router.post("/evaluate_summary")
async def evaluate_summary(
    request: SummaryEvaluationRequest,
) -> SummaryEvaluationResponse:

    statistics_list = await wrapper(
        summary_model_name_list=request.summary_model_name_list,
        summary_prompt_template_list=request.summary_prompt_template_list,
        evaluate_summary_model_name_list=request.evaluate_summary_model_name_list,
        evaluate_summary_sentence_prompt_template_list=request.evaluate_summary_sentence_prompt_template_list,
        documents=request.documents,
    )

    response = SummaryEvaluationResponse(model_metrics_list=statistics_list)
    print(response)
    return response
