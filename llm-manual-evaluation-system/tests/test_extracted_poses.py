import requests
import json
import os

url = "http://localhost:8080/evaluate/evaluate_extracted_poses"


with open(
    os.path.join("statistics", "extracted_poses", "extracted_poses_statistics.json"),
    "r",
) as infile:
    extracted_poses_statistics_list = json.load(infile)

with open(
    os.path.join("datasets", "extracted_poses", "extracted_poses_test_dataset.json"),
    "r",
) as infile:
    extracted_poses_evaluation_set = json.load(infile)


extracted_poses_prompt_template = """
                Based on the following dialogue between Person_A and Person_B: $dialogue
                I want you to create a dictionary with
                the following format:
                    {
                    "Person_A": str,
                    "Person_B": str
                    },
                where each string represents a pose assigned from the list ["sitting", "standing", "kneeling", "laying down", "None"]. The pose is assigned by analyzing the content of the dialogue
                """

requestBody = {
    "model_name_list": ["meta-llama/llama-3-8b-instruct"],
    "prompt_template_list": [extracted_poses_prompt_template],
    "extracted_poses_evaluation_set": extracted_poses_evaluation_set[:3],
}

response = requests.post(url, json=requestBody)
print(response.text)


extracted_poses_statistics_list.append(json.loads(response.text))

with open(
    os.path.join("statistics", "extracted_poses", "extracted_poses_statistics.json"),
    "w",
) as outfile:
    outfile.write(json.dumps(extracted_poses_statistics_list))
