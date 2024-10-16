import os
import re
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crew_ai import CrewAI


credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "*******"
}

project_id = "********"


pdf_folder = "data"
documents = []


if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"The folder '{pdf_folder}' does not exist.")

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        documents += data

if not documents:
    raise ValueError("No PDF files found in the folder.")


doc_id = 0
for doc in documents:
    doc.page_content = " ".join(
        doc.page_content.split())
    doc.metadata["id"] = doc_id
    doc_id += 1


text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
)


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)


retriever = vectorstore.as_retriever()


model_id = ModelTypes.GRANITE_13B_CHAT_V2

parameters = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.TEMPERATURE: 2,
    GenParams.TOP_P: 0,
    GenParams.TOP_K: 100,
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.REPETITION_PENALTY: 1.2,
    GenParams.STOP_SEQUENCES: ['B)', '\n'],
    GenParams.RETURN_OPTIONS: {
        'input_tokens': True, 'generated_tokens': True, 'token_logprobs': True, 'token_ranks': True,
    }
}

llm = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)


quiz_template = """You are an AI tutor. Your task is to prepare a quiz from the following content:

{context}

Create a quiz with multiple-choice questions based on this content to the student to check his knowledge.
"""

report_template = """You are an AI report generator. Your task is to create a student quiz performance report. The student's answers and the correct answers are as follows:

Student Answers: {student_answers}
Correct Answers: {correct_answers}

Generate a detailed report of the student's performance, including areas of improvement and strengths.
"""

answer_template = """You are an AI tutor designed to teach students from diverse backgrounds and learning levels. Your role is to provide personalized explanations of subjects based on the student's grade, learning style, and pace. The goal is to make complex concepts easy to understand by relating them to real-life examples that the student can see or experience. Always adjust your tone to be friendly, encouraging, and supportive. Based on the following content:

{context}

Answer the student's question:
{question}

Provide a detailed and helpful response.
"""

quiz_prompt = ChatPromptTemplate.from_template(quiz_template)
report_prompt = ChatPromptTemplate.from_template(report_template)
answer_prompt = ChatPromptTemplate.from_template(answer_template)


crew = CrewAI()


quiz_agent = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | quiz_prompt
    | llm
    | StrOutputParser()
)


report_agent = (
    {"student_answers": RunnablePassthrough(), "correct_answers": RunnablePassthrough()}
    | report_prompt
    | llm
    | StrOutputParser()
)


answer_agent = (
    {"context": retriever, "question": RunnablePassthrough()}
    | answer_prompt
    | llm
    | StrOutputParser()
)


correct_answers = []


def sanitize_input(input_text):
    return re.sub(r'[\n\t]', ' ', input_text).strip()

# Helper functions


def extract_correct_answers(quiz_response):

    correct_answers_list = []

    return correct_answers_list


def extract_student_answers(user_message):

    student_answers_list = []

    return student_answers_list


def main():
    global correct_answers
    while True:
        user_message = input("Enter your query (or 'exit' to quit): ").strip()

        if user_message.lower() == "exit":
            print("Goodbye!")
            break

        try:
            if "prepare quiz" in user_message.lower():

                response = quiz_agent.invoke(user_message)
                correct_answers = extract_correct_answers(response)
                print(f"\nQuiz Generated:\n{response}\n")

            elif "generate report" in user_message.lower():

                student_answers = extract_student_answers(user_message)
                if not correct_answers:
                    print("No quiz has been prepared yet to generate a report.")
                else:

                    response = report_agent.invoke(
                        {"student_answers": student_answers, "correct_answers": correct_answers})
                    print(f"\nReport Generated:\n{response}\n")

            else:

                response = answer_agent.invoke({"question": user_message})
                print(f"\nAnswer:\n{response}\n")

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
