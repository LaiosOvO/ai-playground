import json
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveJsonSplitter
from pymilvus import MilvusClient
from tqdm import tqdm


_ = load_dotenv(".env")


# 灌库
class ChatDoc(object):

    def __init__(self):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".csv": CSVLoader,
            ".json": self.handle_json,
        }

        self.txt_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=30, length_function=len,
                                                  add_start_index=True)
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=240)
        self.embeding = OpenAIEmbeddings(model="text-embedding-3-small",base_url="https://api.gptsapi.net/v1")
        # self.milvus_client = MilvusClient(host="47.116.119.179", port=19530)
        self.milvus_client = MilvusClient(uri="http://122.51.107.214:19530",dimension=1536)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o",base_url="https://api.gptsapi.net/v1")
        # self.end_splitter = self.txt_splitter.split_documents("")

    def get_knowledge_type(self, filename):
        system_prompt = """
            你是一名知识分类专家，主要分别判断以下类别的知识，有且仅有空调，电视机，冰箱这三类知识。识别准确后，返回给用户。
            识别到空调，返回air_conditioning，识别到冰箱，返回refrigerator，识别到电视，返回TV。
            """
        grade_messages = [SystemMessage(content=system_prompt)]
        data = self.get_file(filename)[0].page_content
        grade_messages.append(HumanMessage(content=f"{data}"))
        collection_name = self.llm.invoke(grade_messages)

        return collection_name.content.strip()

    def get_file(self, filename):
        file_extension = os.path.splitext(filename)[-1]
        loader = self.loader.get(file_extension, None)
        if loader:
            if file_extension == ".json":
                return loader(filename)
            else:
                load_info = loader(filename).load()
                return load_info

        else:
            return None

    def handle_json(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read()
        return data

    def is_json(self, data):
        try:
            json.loads(data)
            return True
        except:
            return False

    def split_text(self, filename):
        load_info = self.get_file(filename)
        if load_info:
            if self.is_json(load_info):
                self.end_splitter = self.json_splitter.split_text(json.loads(load_info), ensure_ascii=False)
            else:
                self.end_splitter = self.txt_splitter.split_documents(load_info)

            return self.end_splitter

        else:
            return "文件格式不支持"

    def emb_text(self, text):
        return self.embeding.embed_query(text)

    def vector_storage(self, filename):
        self.end_splitter = self.split_text(filename)
        data_name = self.get_knowledge_type(filename)
        data = []
        for idx, text in enumerate(tqdm(self.end_splitter, desc="向量化")):
            if isinstance(text, Document):
                text = text.page_content
            data.append({"id": idx, "vector": self.emb_text(text), "text": text})
        try:
            self.milvus_client.create_collection(
                collection_name=data_name,
                dimension=1536,
                metric_type="IP",  # Inner product distance
                consistency_level="Strong",  # Strong consistency level
             )
        except:
            print("已经有对应的表格")

        self.milvus_client.insert(collection_name=data_name, data=data)
        return "向量存储成功"



# if __name__ == "__main__":
#     doc = ChatDoc()
#     doc.vector_storage("./空调.docx")