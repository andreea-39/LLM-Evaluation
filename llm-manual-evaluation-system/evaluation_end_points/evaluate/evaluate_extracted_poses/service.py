import json
import re
from string import Template
import time
from typing import Any, Dict, List

import aiohttp

from evaluation_end_points.evaluate.evaluate_extracted_poses.schema import (
    ModelMetrics,
    ModelName,
    PromptTemplate,
)
from evaluation_end_points.settings import get_settings


settings = get_settings()


async def generate_prediction(dialogue: str, model_name: str, prompt_template: str):
    request_data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": Template(prompt_template).substitute(
                    dialogue=dialogue,
                ),
            }
        ],
        "top_p": 1,
        "top_k": 0,
        "repetition_penalty": 1,
        "temperature": 0.7,
        "max_tokens": 300,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=request_data,
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        ) as response:
            response_json = json.loads(await response.text())

    while (
        "error" in response_json.keys()
        and re.search(
            "\{.+\}",
            re.sub("\n", "", response_json["choices"][0]["message"]["content"].strip()),
        )
        == None
    ):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            ) as response:
                response_json = json.loads(await response.text())

        time.sleep(10) if "free" in model_name else time.sleep(0)

    output = response_json["choices"][0]["message"]["content"].strip()
    output = json.loads(re.search("\{.+\}", re.sub("\n", "", output)).group().strip())

    print(output)

    return output


def compute_number_of_correct_and_wrong_predictions(
    predicted_values_list, actual_values_list
):

    correct_predictions = 0
    wrong_predictions = 0

    for index in range(len(predicted_values_list)):
        if (
            predicted_values_list[index]["Person_A"]
            == actual_values_list[index].Person_A
        ):
            correct_predictions += 1
        else:
            wrong_predictions += 1
        if (
            predicted_values_list[index]["Person_B"]
            == actual_values_list[index].Person_B
        ):
            correct_predictions += 1
        else:
            wrong_predictions += 1

    return correct_predictions, wrong_predictions


def compute_metrics(predicted_values_list, actual_values_list):

    correct_predictions, wrong_predictions = (
        compute_number_of_correct_and_wrong_predictions(
            predicted_values_list=predicted_values_list,
            actual_values_list=actual_values_list,
        )
    )

    accuracy = correct_predictions / (correct_predictions + wrong_predictions)

    return accuracy


async def wrapper(
    model_name_list: List[ModelName],
    prompt_template_list: List[PromptTemplate],
    extracted_poses_evaluation_set: List[Any],
):
    statistics_list = []
    actual_values_list = []
    predicted_values_list = []

    dialogue_analysis = []

    for model_name in model_name_list:
        for prompt_template in prompt_template_list:
            for index in range(len(extracted_poses_evaluation_set)):

                dialogue = extracted_poses_evaluation_set[index].dialogue
                actual_poses = extracted_poses_evaluation_set[index].extracted_poses
                actual_values_list.append(actual_poses)

                predicted_poses = await generate_prediction(
                    dialogue=dialogue,
                    model_name=model_name,
                    prompt_template=prompt_template,
                )

                predicted_values_list.append(predicted_poses)

                dialogue_analysis.append(
                    {
                        "dialogue": dialogue,
                        "actual_poses": actual_poses,
                        "predicted_poses": predicted_poses,
                    }
                )

            accuracy = compute_metrics(predicted_values_list, actual_values_list)
            statistics_dictionary = ModelMetrics(
                model_name=model_name,
                prompt_template=prompt_template,
                accuracy=accuracy,
                dialogues_analysis=dialogue_analysis,
            )
            statistics_list.append(statistics_dictionary)
    return statistics_list
