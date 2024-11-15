from fastapi import FastAPI, File, UploadFile,Query

from pydantic import BaseModel, Field

from base_tool import ChatDoc
from create_graph_teach import CreateLLMCustomerService


app = FastAPI()


chat_rag =  CreateLLMCustomerService()

class questionInput(BaseModel):
    question: str = Field(..., description="问题")


@app.post("/chat_file")
async def chat_file(question:str=Query(None),file:UploadFile=File(None)):
    if file:
        contents = await file.read()
        file_name = file.filename
        with open(file_name, "wb") as f:
            f.write(contents)
        data = {"question": question, "filename": file_name}
    else:
        data = {"question": question,"filename":None}

    res = CreateLLMCustomerService().chat(**data)
    return res


@app.post("/vector")
async def vector(file:UploadFile=File(None)):
    contents = await file.read()
    file_name = file.filename
    with open(file_name, "wb") as f:
        f.write(contents)
    insert_vector = ChatDoc()
    insert_vector.split_text(filename=file_name)
    result = insert_vector.vector_storage(filename=file_name)
    return result



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=5000)
