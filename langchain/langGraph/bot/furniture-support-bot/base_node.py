import concurrent.futures

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from self_rag_tool import GradeAndGenerateTool
from vector_storage import VectorStorageObject
from dotenv import load_dotenv

_ = load_dotenv(".env")
tools = GradeAndGenerateTool()
vector_tool = VectorStorageObject()


def get_knowledge_type(state):
    question = state["question"]
    filename = state["filename"]
    system_prompt = """
                你是一名知识分类专家，主要分别判断以下类别的知识，有且仅有空调，电视机，冰箱这三类知识。识别准确后，返回给用户。
                识别到空调，返回air_conditioning，识别到冰箱，返回refrigerator，识别到电视，返回TV。
                """
    grade_messages = [SystemMessage(content=system_prompt)]
    grade_messages.append(HumanMessage(content=f"{question}"))
    llm = ChatOpenAI(temperature=0, model="gpt-4o",base_url="https://api.gptsapi.net/v1")
    res = llm.invoke(grade_messages)
    return {"question": question, "collection_name": res.content.strip(), "filename": filename}


def retrieve(state):
    question = state["question"]
    collection_name = state["collection_name"]
    filename = state["filename"]
    if filename:
        documents = vector_tool.split_text(filename)
    else:
        documents = tools.search_vector(question, collection_name)
    result_documents = []
    for document in documents[0]:
        result_documents.append(document["entity"]["text"])
    return {"question": question, "documents": result_documents, "collection_name": collection_name,
            "filename": filename}


# 评分 搜索文档和用户问题的相关性 将yes的文档过滤出来
def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filename = state["filename"]
    grade_count = state["grade_count"]
    filtter_documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(tools.grade, question, document, documents.index(document)) for document in
                   documents}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result[0] == "yes":
                filtter_documents.append(documents[result[1]])
    if grade_count:
        grade_count += 1
    else:
        grade_count = 1

    return {"question": question, "documents": filtter_documents, "filename": filename, "grade_count": grade_count}


def generation(state):
    question = state["question"]
    documents = state["documents"]
    hallucination_count = state["hallucination_count"]
    if hallucination_count:
        hallucination_count += 1
    else:
        hallucination_count = 1
    document = "\n".join(documents).replace("{", "").replace("}", "")
    answer = tools.generate(question, document)
    return {"question": question, "answer": answer, "hallucination_count": hallucination_count}


def rewrite_question(state):
    question = state["question"]
    rewrite_question = tools.rewrite_question(question)
    return {"question": rewrite_question}


def grade_generation(state):
    documents = state["documents"]
    filename = state["filename"]
    grade_count = state["grade_count"]
    if grade_count > 3:
        return "end_answer"
    else:
        if documents:
            return "generation"
        else:
            if filename:
                return "file_out"
            return "rewrite_question"


def hallucinations_generate(state):
    documents = state["documents"]
    answer = state["answer"]
    question = state["question"]
    hallucination_count = state["hallucination_count"]
    hallucinations_score = tools.hallucinations(documents, answer)
    if hallucination_count > 3:
        return "end_answer"
    if hallucinations_score == "yes":
        return "generation"
    else:
        answer_score = tools.answer_question(question, answer)
        if answer_score == "yes":
            return "useful"
        else:
            return "rewrite_question"


def vector_storage(state):
    question = state["question"]
    filename = state["filename"]
    documents = vector_tool.split_text(filename)
    return {"question": question, "documents": documents}


def end_answer(state):
    answer  = "对不起我暂时无法回复该问题"
    return {"answer":answer}

def file_out(state):
    answer = "你提供的文档不支持回答"
    return {"answer":answer}


def route_node(state):
    filename = state["filename"]
    if filename:
        return "vector_storage"
    else:
        return "get_knowledge_type"
