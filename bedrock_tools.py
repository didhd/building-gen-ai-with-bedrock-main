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
    name = "well_architected_tool"
    description = "Use this tool for any AWS related question to help customers understand best practices on building on AWS. It will use the relevant context from the AWS Well-Architected Framework to answer the customer's query. The input is the customer's question. The tool returns an answer for the customer using the relevant context."
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query):
        # Find docs
        embeddings = BedrockEmbeddings()
        vectorstore = FAISS.load_local("local_index", embeddings)
        docs = vectorstore.similarity_search(query)
        context = ""

        doc_sources_string = ""
        for doc in docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end.

        {context}

        Question: {query}
        Answer:"""

        generated_text = call_bedrock(prompt)
        print(generated_text)

        resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json




class CodeGenerationTool(Tool):
    name = "code_generation_tool"
    description = "Use this tool only when you need to generate code based on a customers's request. The input is the customer's question. The tool returns code that the customer can use."

    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, prompt):
        generated_text = call_bedrock(prompt)
        return generated_text


def get_embedding(body, modelId, accept, contentType):
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")
    return embedding


class InternetQueryTool(Tool):
    name = "internet_query_tool"
    description = "Use this tool to query certain document on Internet."
    inputs = ["text"]
    outputs = ["text"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)

    # Load Wikipedia  - 위키피디아에서 자료 가져오기
    wikipedia_loader = WikipediaLoader(query="AWS", load_max_docs=2)
    wikipedia_texts = wikipedia_loader.load_and_split(text_splitter=text_splitter)

    # Load URLs - 인터넷 웹페이지 크롤링 - Amazon Rekognition 온라인 문서 추출
    urls = [
        "https://docs.aws.amazon.com/rekognition/latest/dg/labels.html",
        "https://docs.aws.amazon.com/rekognition/latest/dg/faces.html",
        "https://docs.aws.amazon.com/rekognition/latest/dg/collections.html",
        "https://docs.aws.amazon.com/rekognition/latest/dg/celebrities.html",
    ]
    url_loader = UnstructuredURLLoader(urls=urls)
    url_texts = url_loader.load_and_split(text_splitter=text_splitter)

    # Load PDF - PDF 소스 활용 - RAG 논문
    sagemaker_pdf_url = "https://arxiv.org/pdf/2005.11401"
    response = requests.get(sagemaker_pdf_url)
    with open(f"/tmp/rag_paper.pdf", "wb") as file:
        file.write(response.content)
    pdf_loader = PDFPlumberLoader(f"/tmp/rag_paper.pdf")
    pdf_texts = pdf_loader.load_and_split(text_splitter=text_splitter)

    # Build vector index
    all_texts = wikipedia_texts + pdf_texts + url_texts
    embeddings = BedrockEmbeddings()
    agg_docsearch = FAISS.from_documents(all_texts, embeddings)

    def __call__(self, query):
        res_docs = self.agg_docsearch.similarity_search(query, k=5)

        context = ""
        doc_sources_string = ""
        for doc in res_docs:
            doc_sources_string += doc.metadata["source"] + "\n"
            context += doc.page_content

        prompt = f"""Use the following pieces of context to answer the question at the end.

        {context}

        Question: {query}
        Answer:"""

        generated_text = call_bedrock(prompt)
        print(generated_text)

        resp_json = {"ans": str(generated_text), "docs": doc_sources_string}
        return resp_json

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
