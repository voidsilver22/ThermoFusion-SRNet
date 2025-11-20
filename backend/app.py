from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
import traceback

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the output directory to serve static files
os.makedirs("output_results", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="output_results"), name="outputs")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_input_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        output_dir = "output_results"
        os.makedirs(output_dir, exist_ok=True)

        # Add parent directory to path to import Model.inference
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        
        # Import here to catch import errors during the request
        from Model.inference import run_inference
        
        print(f"Running inference on {temp_input_path}...")
        result = run_inference(temp_input_path, output_dir)
        
        if "error" in result:
            return result

        # Convert absolute paths to URLs
        base_url = "http://127.0.0.1:8000/outputs"
        result["input_preview"] = f"{base_url}/{os.path.basename(result['input_preview'])}"
        result["output_preview"] = f"{base_url}/{os.path.basename(result['output_preview'])}"
        result["output_plot"] = f"{base_url}/{os.path.basename(result['output_plot'])}"
        
        return result
            
    except ImportError as e:
        print(f"Import Error: {e}")
        traceback.print_exc()
        return {
            "error": f"Import Error (GDAL/Dependencies): {str(e)}",
            "traceback": traceback.format_exc()
        }
    except Exception as e:
        print(f"Runtime Error: {e}")
        traceback.print_exc()
        return {
            "error": f"Runtime Error: {str(e)}",
            "traceback": traceback.format_exc()
        }
    finally:
        # Cleanup temp file
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except:
                pass
