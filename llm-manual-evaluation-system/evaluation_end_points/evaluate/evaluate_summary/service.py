import json
import re
from string import Template
import time
from typing import List

import aiohttp

from evaluation_end_points.evaluate.evaluate_summary.schema import (
    Document,
    ModelName,
    PromptTemplate,
)


from evaluation_end_points.settings import get_settings

from sentence_transformers import SentenceTransformer
import faiss

settings = get_settings()
sentence_transformer_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

PHRASES_TO_ELIMINATE = [
    "Here is a concise summary of the text:",
    "The summary of the text is:",
    "I've analyzed the text and condensed it to \d+ tokens or less.",
    "Here's the concise summary:",
    "Here is a concise summary of the text in \d+ tokens or less:",
    "Here is a concise summary of the text within the \d+-token limit:",
    "Here is a concise summary of the text in \d+ tokens",
    "Here is a concise summary of the text (within the \d+-token limit):",
    "Here is a concise summary of the provided text in \d+ tokens or less:",
]


async def splitTextIntoChunks(text: str, chunk_size=15000) -> List[str]:
    chunks = []
    for index in range(0, len(text), chunk_size):
        if index == 0:
            chunk = text[0:chunk_size]
        else:
            chunk = text[
                index
                - (int(chunk_size / 3)) : min(
                    index + (int(2 * chunk_size / 3)), len(text)
                )
            ]
        chunks.append(chunk)
    return chunks


async def eliminate_stop_phrases(
    summary: str, stop_phrases: List[str] = PHRASES_TO_ELIMINATE
) -> str:
    for phrase in stop_phrases:
        summary = re.sub(phrase, "", summary)
    return summary


async def preprocess_summary(summary: str) -> str:
    summary = await eliminate_stop_phrases(summary=summary)
    return summary


async def generate_summary(text: str, model_name: str, prompt_template: str) -> str:
    request_data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": Template(prompt_template).substitute(
                    text=text,
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

    print(response_json)

    while "error" in response_json.keys():
        # print(response_json)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            ) as response:
                response_json = json.loads(await response.text())
        time.sleep(10) if "free" in model_name else time.sleep(0)

    output = response_json["choices"][0]["message"]["content"].strip()
    return output


async def generate_final_summary(
    document_text: str, model_name: str, prompt_template: str
) -> str:
    chunks = await splitTextIntoChunks(document_text)

    summaries_list = []
    for index, chunk in enumerate(chunks):
        print("Generating temporary summary ", index + 1, "/", len(chunks))
        summary = await generate_summary(
            text=chunk, model_name=model_name, prompt_template=prompt_template
        )
        summary = await preprocess_summary(summary)
        summaries_list.append(summary)

        time.sleep(10) if "free" in model_name else time.sleep(0)

    chunk_summaries_text = ""
    for summary in summaries_list:
        chunk_summaries_text += " " + summary

    print("Generating final summary")
    final_summary = await generate_summary(
        text=chunk_summaries_text,
        model_name=model_name,
        prompt_template=prompt_template,
    )
    final_summary = await preprocess_summary(final_summary)
    return final_summary


async def create_document_embeddings_and_index(chunks: List[str]):

    document_embeddings = sentence_transformer_model.encode(chunks)

    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)

    return document_embeddings, index


async def return_retrieved_documents(
    chunks: List[str], faiss_index: faiss.IndexFlatL2, query: str, k: int = 4
):
    query_embeddings = sentence_transformer_model.encode([query])

    distances, indices = faiss_index.search(query_embeddings, k=k)
    retrieved_documents = [chunks[index] for index in indices[0]]

    return retrieved_documents


async def test_summary_sentence_validity(
    evaluate_summary_model_name: str,
    summary_sentence_verification_prompt: str,
    sentence: str,
    context: str,
) -> bool:

    request_data = {
        "model": evaluate_summary_model_name,
        "messages": [
            {
                "role": "system",
                "content": Template(summary_sentence_verification_prompt).substitute(
                    sentence=sentence, context=context
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

    while "error" in response_json.keys():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            ) as response:
                response_json = json.loads(await response.text())
                (
                    time.sleep(10)
                    if "free" in evaluate_summary_model_name
                    else time.sleep(0)
                )

    response = response_json["choices"][0]["message"]["content"]

    if "true" in response.lower():
        return True
    return False


async def test_summary(
    document_text: str,
    summary: str,
    evaluate_summary_model_name: str,
    summary_sentence_verification_prompt: PromptTemplate,
):
    correct_sentences = 0
    wrong_sentences = 0
    chunks = await splitTextIntoChunks(document_text)
    document_embeddings, faiss_index = await create_document_embeddings_and_index(
        chunks
    )
    summary_sentences_statistics = []

    for sentence_index, sentence in enumerate(
        summary.split(".")[: len(summary.split(".")) - 1]
    ):

        print(f"Evaluating sentence {sentence_index+1}/{len(summary.split('.')) - 1}")

        query = sentence
        sentence_context_matches = []
        query_embeddings = sentence_transformer_model.encode([query]).astype("float32")
        distances, indices = faiss_index.search(query_embeddings, 4)

        has_at_least_one_true = False

        for document_index in indices[0]:
            context = chunks[document_index]
            validity = await test_summary_sentence_validity(
                evaluate_summary_model_name,
                summary_sentence_verification_prompt,
                sentence=sentence,
                context=context,
            )

            if validity == True:
                has_at_least_one_true = True
                sentence_context_matches.append({"context": context, "validity": True})
            else:
                sentence_context_matches.append({"context": context, "validity": False})

        if has_at_least_one_true == True:
            correct_sentences += 1
        else:
            wrong_sentences += 1

        summary_sentences_statistics.append(
            {"sentence": sentence, "sentence_context_matches": sentence_context_matches}
        )

    return correct_sentences, wrong_sentences, summary_sentences_statistics


async def wrapper(
    summary_model_name_list: list[ModelName],
    summary_prompt_template_list: List[PromptTemplate],
    evaluate_summary_model_name_list: list[ModelName],
    evaluate_summary_sentence_prompt_template_list: List[PromptTemplate],
    documents: List[Document],
):
    summaries_list = []
    for document in documents:
        print("Started summaries generation for PDF: ", document.document_title)
        for summary_model_name in summary_model_name_list:
            for summary_prompt_template in summary_prompt_template_list:
                summary = await generate_final_summary(
                    document.document_text, summary_model_name, summary_prompt_template
                )

                print(
                    f"Ended summary generation for PDF: {document.document_title}, with model {summary_model_name}"
                )

                print(f"Started evaluating summary")
                for evaluate_summary_model_name in evaluate_summary_model_name_list:
                    for (
                        evaluate_summary_sentence_prompt_template
                    ) in evaluate_summary_sentence_prompt_template_list:

                        (
                            correct_sentences,
                            wrong_sentences,
                            summary_sentences_statistics,
                        ) = await test_summary(
                            document.document_text,
                            summary,
                            evaluate_summary_model_name,
                            evaluate_summary_sentence_prompt_template,
                        )
                print(f"Ended evaluating summary")
                summary_dictionary = {
                    "summary_model_name": summary_model_name,
                    "summary_prompt_template": summary_prompt_template,
                    "document_title": document.document_title,
                    "summary": summary,
                    "evaluate_summary_model_name": evaluate_summary_model_name,
                    "evaluate_summary_sentence_prompt_template": evaluate_summary_sentence_prompt_template,
                    "correct_sentences": correct_sentences,
                    "wrong_sentences": wrong_sentences,
                    "summary_sentences_statistics": summary_sentences_statistics,
                }
                summaries_list.append(summary_dictionary)

    return summaries_list
