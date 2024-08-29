import requests
import json
import os

from service import load_PDF_text, reduce_extra_spaces

url = "http://localhost:8080/evaluate/evaluate_summary"


with open(
    os.path.join("statistics", "summary", "summary_statistics.json"), "r"
) as infile:
    summary_statistics_list = json.load(infile)


documents_list = []

for file_name in os.listdir(os.path.join("datasets", "summary", "documents")):
    file_path = os.path.join("datasets", "summary", "documents", file_name)
    file_text = load_PDF_text(file_path)
    file_text = reduce_extra_spaces(file_text)

    documents_list.append({"document_title": file_name, "document_text": file_text})


summary_prompt_template = """
    Write a concise summary of the following text delimited by triple backquotes.

    ```$text```

    CONCISE SUMMARY:"""

evaluate_summary_sentence_prompt_template = """
    Print true if the information written in the query text is related to the information in the context text. Otherwise print false. Don't write any python code, just print true or false
    query: $sentence,
    context: $context
"""


request_body = {
    "summary_model_name_list": ["meta-llama/llama-3-8b-instruct"],
    "summary_prompt_template_list": [summary_prompt_template],
    "evaluate_summary_model_name_list": ["meta-llama/llama-3-8b-instruct"],
    "evaluate_summary_sentence_prompt_template_list": [
        evaluate_summary_sentence_prompt_template
    ],
    "documents": documents_list,
}

response = requests.post(url, json=request_body)
print(response.text)


summary_statistics_list.append(json.loads(response.text))

with open(
    os.path.join("statistics", "summary", "summary_statistics.json"), "w"
) as outfile:
    outfile.write(json.dumps(summary_statistics_list))
