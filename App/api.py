# pip install fastapi
# pip install uvicorn
# pip install python-multipart

from fastapi import FastAPI, File
from typing import Dict
import App.caption as caption
import json

app = FastAPI()

@app.get("/health")
def Health():
    return {"Status": "Running"}

@app.post("/caption")
def get_image(file: bytes = File()):
    result = caption.evaluate(file)
    result = ' '.join(result)
    return json.dumps(result)

# if image path is passed instead of the file:

# @app.post("/caption")
# def get_image(img_path: Dict):
#     img_path = img_path.get("path")
#     result = caption.evaluate(str(img_path))
#     result = ' '.join(result)
#     return json.dumps(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8001)

# "localhost"
