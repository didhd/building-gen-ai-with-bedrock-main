import json

import boto3
import requests
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from transformers import Tool
from langchain.document_loaders import (
    WikipediaLoader,
    UnstructuredURLLoader,
    PDFPlumberLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


bedrock = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    endpoint_url="https://bedrock.us-east-1.amazonaws.com",
)


def call_bedrock(prompt, maxTokenCount=4096, temperature=0.5, topP=0.2):
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": maxTokenCount,
            "stopSequences": [],
            "temperature": temperature,
            "topP": topP,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-tg1-large"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results


class AWSWellArchTool(Tool):
    # Your Code Herer
    pass


class CodeGenerationTool(Tool):
    # Your Code Herer
    pass


def get_embedding(body, modelId, accept, contentType):
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")
    return embedding


class InternetQueryTool(Tool):
    # Your Code Herer
    pass


#### Testing Well Architected Tool
# query = "How can I design secure VPCs?"
# well_arch_tool = AWSWellArchTool()
# output = well_arch_tool(query)
# print(output)


#### Testing Code Generation Tool
# query = "Write a function in Python to upload a file to Amazon S3"
# code_gen_tool = CodeGenerationTool()
# output = code_gen_tool(query)
# print(output)
