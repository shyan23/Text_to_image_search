from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from image_process import ImageMetadataProcessor, ImageRetriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  

app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount public folder for image access
os.makedirs("public", exist_ok=True)
app.mount("/public", StaticFiles(directory="public"), name="public")

class ProcessRequest(BaseModel):
    image_paths: List[str]

class QueryRequest(BaseModel):
    query: str

# Global variables
processor = ImageMetadataProcessor()
retriever = None

@app.get("/")
async def root():
    return {"message": "Image Processing API is running"}

@app.post("/process_images/")
async def process_images(request: ProcessRequest):
    global retriever
    
    try:
        # Validate image paths
        valid_paths = []
        for path in request.image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"Warning: Image path does not exist: {path}")
        
        if not valid_paths:
            raise HTTPException(status_code=400, detail="No valid image paths provided")
        
        # Process images and create retriever
        metadata_store = processor.process_images(valid_paths)
        if not metadata_store:
            raise HTTPException(status_code=400, detail="No metadata could be processed from images")
        
        # Create retriever with processed metadata
        retriever = ImageRetriever(metadata_store)
        retriever.create_vector_store()  # Initialize the vector store
        
        return {
            "message": "Images processed successfully",
            "processed_count": len(metadata_store),
            "metadata": metadata_store
        }
    
    except Exception as e:
        print(f"Error in process_images: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query_images/")
async def query_images(request: QueryRequest):
    global retriever
    
    try:
        if not retriever:
            raise HTTPException(status_code=400, detail="Process images first using /process_images/ endpoint")
        
        # Use the simple_search method which returns properly formatted results
        results = retriever.simple_search(request.query, limit=5)
        
        return {
            "query": request.query,
            "results": results
        }
    
    except Exception as e:
        print(f"Error in query_images: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/")
async def get_status():
    return {
        "processor_initialized": processor is not None,
        "retriever_initialized": retriever is not None,
        "metadata_count": len(processor.metadata_store) if processor else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)