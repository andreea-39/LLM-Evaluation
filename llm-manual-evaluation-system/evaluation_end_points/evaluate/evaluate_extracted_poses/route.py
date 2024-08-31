from evaluation_end_points.evaluate.evaluate_extracted_poses.service import wrapper
from evaluation_end_points.evaluate.evaluate_extracted_poses.schema import (
    ExtractedPosesEvaluationRequest,
    ExtractedPosesEvaluationResponse,
)
from ..router import router


@router.post("/evaluate_extracted_poses")
async def evaluate_story_objective(
    request: ExtractedPosesEvaluationRequest,
) -> ExtractedPosesEvaluationResponse:

    statistics_list = await wrapper(
        model_name_list=request.model_name_list,
        prompt_template_list=request.prompt_template_list,
        extracted_poses_evaluation_set=request.extracted_poses_evaluation_set,
    )

    response = ExtractedPosesEvaluationResponse(model_metrics_list=statistics_list)
    print(response)
    return response
