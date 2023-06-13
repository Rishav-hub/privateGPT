from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
import os
from fastapi import FastAPI, UploadFile, File
from typing import List, Optional
import urllib.parse
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from torch import cuda, bfloat16 



load_dotenv()

embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"
model = "tiiuae/falcon-7b-instruct"


# model_type = os.environ.get('MODEL_TYPE')
# model_path = os.environ.get('MODEL_PATH')
# model_n_ctx = os.environ.get('MODEL_N_CTX')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

from constants import CHROMA_SETTINGS

# async def test_embedding():
#     # Create the folder if it doesn't exist
#     os.makedirs(source_directory, exist_ok=True)
#     # Create a sample.txt file in the source_documents directory
#     file_path = os.path.join("source_documents", "test.txt")
#     with open(file_path, "w") as file:
#         file.write("This is a test.")
#     # Run the ingest.py command
#     os.system('python ingest.py --collection test')
#     # Delete the sample.txt file
#     os.remove(file_path)
#     print("embeddings working")

# async def model_download():
#     match model_type:
#         case "LlamaCpp":
#             url = "https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
#         case "GPT4All":
#             url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
#     folder = "models"
#     parsed_url = urllib.parse.urlparse(url)
#     filename = os.path.join(folder, os.path.basename(parsed_url.path))
#     # Check if the file already exists
#     if os.path.exists(filename):
#         print("File already exists.")
#         return
#     # Create the folder if it doesn't exist
#     os.makedirs(folder, exist_ok=True)
#     # Run wget command to download the file
#     os.system(f"wget {url} -O {filename}")
#     global model_path 
#     model_path = filename
#     os.environ['MODEL_PATH'] = filename
#     print("model downloaded")
    

# Starting the app with embedding and llm download
# @app.on_event("startup")
# async def startup_event():
#     await test_embedding()
#     await model_download()


# # Example route
# @app.get("/")
# async def root():
#     return {"message": "Hello, the APIs are now ready for your embeds and queries!"}

def embed_documents(files, collection_name: Optional[str] = None):

    saved_files = []
    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(source_directory, file.filename)
        saved_files.append(file_path)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        if collection_name is None:
            # Handle the case when the collection_name is not defined
            collection_name = file.filename
    
    os.system(f'python ingest.py --collection {collection_name}')
    
    # Delete the contents of the folder
    [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file in files]
    
    return {"message": "Files embedded successfully", "saved_files": saved_files}

def retrieve_documents(query: str, collection_name:str):
    target_source_chunks = 4
    mute_stream = ""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory,collection_name=collection_name, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # Prepare the LLM
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]

    llm = HuggingFacePipeline.from_model_id(model_id=model, task="text-generation", device=0, model_kwargs={"temperature":0.1,"trust_remote_code": True, "max_length":100000, "top_p":0.15, "top_k":0, "repetition_penalty":1.1, "num_return_sequences":1,})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    
    # Get the answer from the chain
    res = qa(query)
    print(res)   
    answer = res['result']

    return {"results": answer}
