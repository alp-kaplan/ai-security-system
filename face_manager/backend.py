from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import cv2
import numpy
from pydantic import BaseModel
from face_manager import FaceDataManager
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files for serving images
app.mount("/images", StaticFiles(directory="/home/alp/face_data/images"), name="images")

# Initialize the FaceDataManager
manager = FaceDataManager()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_frontend():
    """
    Serves the frontend HTML file.

    Returns:
        FileResponse: The frontend HTML file response.
    """
    return FileResponse("frontend.html")


@app.get("/list")
def list_people():
    """
    Lists all people in the face recognition system.

    Returns:
        dict: A dictionary containing a list of people.

    This endpoint retrieves the names of all individuals and returns them as a JSON response.
    """
    _, names = manager.load_existing_data()
    unique_names = sorted(set(names))
    return {"people": unique_names}  # Just a list of strings


@app.get("/list-unknown")
def list_unknown():
    """
    Lists all unknown people detected in the system.

    Returns:
        dict: A dictionary containing a list of unknown persons.

    This endpoint checks the images directory for any unknown persons and returns their names as a JSON response.
    """
    try:
        image_dir = "/home/alp/face_data/images"
        unknowns = [
            d for d in os.listdir(image_dir)
            if d.startswith("unknown_person_") and os.path.isdir(os.path.join(image_dir, d))
        ]
    except Exception as e:
        print(f"Error listing unknowns: {e}")
        unknowns = []
    return {"unknowns": unknowns}


@app.post("/add")
async def add_person(name: str = Form(...), files: List[UploadFile] = None):
    """
    Adds a new person to the face recognition system with uploaded images.

    Args:
        name (str): The name of the person to add.
        files (List[UploadFile]): A list of uploaded image files.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If no files are uploaded or if an invalid image file is provided.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    images = []
    for file in files:
        contents = await file.read()
        np_arr = numpy.frombuffer(contents, numpy.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
        else:
            raise HTTPException(status_code=400, detail="Invalid image file")
    
    manager.add_person(name, images)
    return {"message": f"Added {len(images)} images for {name}"}


@app.delete("/delete/{name}")
def remove_person(name: str):
    """
    Removes all data associated with a specified person.

    Args:
        name (str): The name of the person to remove.

    Returns:
        dict: A message indicating the result of the operation.

    This endpoint deletes the person's data from the encodings file and removes their images.
    """
    manager.remove_person(name)
    return {"message": f"Removed all data of {name}"}


class RenamePersonRequest(BaseModel):
    """
    Pydantic model for renaming a person.

    Attributes:
        old_name (str): The current name of the person to rename.
        new_name (str): The new name to assign to the person.
    """
    old_name: str
    new_name: str

@app.post("/rename")
def rename_person(request: RenamePersonRequest):
    """
    Renames an existing person in the face recognition system.

    Args:
        request (RenamePersonRequest): The request body containing old and new names.

    Returns:
        dict: A message indicating the result of the renaming operation.

    Raises:
        HTTPException: If the renaming fails for any reason.
    """
    try:
        success = manager.rename_person(request.old_name, request.new_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to rename person")
        return {"message": f"Renamed {request.old_name} to {request.new_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/people_with_images")
def people_with_images():
    """
    Retrieves a list of people along with their associated images.

    Returns:
        dict: A dictionary containing a list of people and their images.

    This endpoint returns the names of individuals and a few of their associated images.
    """
    base_dir = "/home/alp/face_data/images"
    people = []

    for name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, name)
        if os.path.isdir(person_dir):
            images = sorted(os.listdir(person_dir))[:3]
            image_paths = [f"/images/{name}/{img}" for img in images]
            people.append({"name": name, "images": image_paths})

    return {"people": people}
