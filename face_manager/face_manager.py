import face_recognition
import os
import pickle
import cv2
import threading
import time
import numpy
import faiss

class FrameCapture:
    """
    FrameCapture class for capturing frames from a video stream.

    This class provides methods for starting and stopping frame capture, 
    retrieving the latest frame, and managing the capture thread.
    """
    def __init__(self, rtsp_url):
        """
        Initializes the FrameCapture with the specified RTSP URL.

        Args:
            rtsp_url (str): The RTSP URL for the video stream.
        """
        self.cap = cv2.VideoCapture(rtsp_url)
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        
    def start(self):
        """
        Starts capturing frames from the video stream in a separate thread.

        This method continuously reads frames from the video stream and 
        updates the latest_frame attribute with the most recent frame.
        """
        def _capture():
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame.copy()
        
        self.thread = threading.Thread(target=_capture, daemon=True)
        self.thread.start()
        
    def get_latest_frame(self):
        """
        Retrieves the latest captured frame.

        Returns:
            numpy.ndarray: The latest frame captured from the video stream, 
                           or None if no frame is available.

        This method uses a lock to ensure thread safety when accessing the latest_frame attribute.
        """
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def stop(self):
        """
        Stops capturing frames and releases the video capture object.

        This method sets the running flag to False, waits for the capture thread to finish, 
        and releases the video capture resource.
        """
        self.running = False
        self.thread.join()
        self.cap.release()


class FaceDataManager:
    """
    FaceDataManager class for managing face data, including capturing, adding, 
    removing, and renaming individuals in the face recognition system.
    """
    def __init__(self,
                 data_file='/home/alp/face_data/encodings.dat',
                 index_file='/home/alp/face_data/faiss.index',
                 names_file='/home/alp/face_data/faiss_names.pkl'):
        """
        Initializes the FaceDataManager with paths for the data files.

        Args:
            data_file (str): Path to the file containing face encodings.
            index_file (str): Path to the FAISS index file.
            names_file (str): Path to the file containing names associated with face encodings.
        """
        self.data_file = data_file
        self.index_file = index_file
        self.names_file = names_file

        self.capture = FrameCapture("rtsp://admin:gsmtest123@10.11.101.100:554/Streaming/channels/101")
        self.capture.start()
        
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        os.makedirs(os.path.dirname(names_file), exist_ok=True)
        
        # Load or initialize FAISS index & names
        self.face_names = []
        self.index = None
        self.load_faiss_index()

    def load_faiss_index(self):
        """
        Loads the FAISS index and associated names from the specified files.

        This method attempts to read the FAISS index and names from their respective files. 
        If loading fails, it initializes the index and names to empty.

        Raises:
            Exception: If there is an error loading the FAISS index or names.
        """
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.names_file, 'rb') as f:
                self.face_names = pickle.load(f)
            print(f"Loaded FAISS index with {len(self.face_names)} faces")
        except Exception as e:
            print(f"FAISS index not found or failed to load: {e}")
            self.index = None
            self.face_names = []

    def save_faiss_index(self, encodings, names):
        """
        Saves the current face encodings and names to the FAISS index and names file.

        Args:
            encodings (list): A list of face encodings to save.
            names (list): A list of names associated with the face encodings.

        This method creates a new FAISS index, adds the encodings, and saves the index 
        and names to their respective files.
        """
        vectors = numpy.array(encodings).astype('float32')
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        faiss.write_index(index, self.index_file)
        with open(self.names_file, 'wb') as f:
            pickle.dump(names, f)
        self.index = index
        self.face_names = names
        print(f"FAISS index saved with {len(names)} faces")
    
    def capture_from_camera(self, name, num_images=3):
        """
        Captures a specified number of face images from the camera.

        Args:
            name (str): The name of the person whose images are being captured.
            num_images (int): The number of images to capture (default is 3).

        This method prompts the user to capture images by pressing ENTER and saves 
        the captured images for the specified person.
        """
        print(f"Capturing {num_images} images for {name}...")
        captured = []

        # Create directory for this person's images
        person_dir = f"/home/alp/face_data/images/{name}"
        os.makedirs(person_dir, exist_ok=True)

        while len(captured) < num_images:
            print("Press ENTER to capture image or type 'exit' to cancel:")
            user_input = input().strip().lower()
            if user_input == 'exit':
                break
            elif user_input == '':
                frame = self.capture.get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                # Calculate current image number
                captured.append(frame)
                print(f"Captured {len(captured)}/{num_images}")

        if captured:
            self.add_person(name, captured)
    
    def add_person(self, name, images):
        """
        Adds a person to the face recognition system using captured images.

        Args:
            name (str): The name of the person to add.
            images (list): A list of images (numpy arrays) of the person.

        This method saves the images to the appropriate directory, extracts face encodings, 
        and updates the data files with the new encodings and names.
        """
        known_encodings, known_names = self.load_existing_data()

        # Ensure both directories exist
        person_dir = f"/home/alp/face_data/images/{name}"
        os.makedirs(person_dir, exist_ok=True)

        # Count existing image files to avoid overwriting
        existing_images = os.listdir(person_dir)
        img_idx = len(existing_images)

        for img in images:
            if isinstance(img, str):  # File path
                image = face_recognition.load_image_file(img)
                # Copy the file to both directories
                base_name = f"{img_idx}.jpg"
                dest = os.path.join(person_dir, base_name)
                # Use cv2 to save as jpg for consistency
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dest, img_bgr)
                
            else:  # numpy array (from camera)
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                base_name = f"{img_idx}.jpg"
                dest = os.path.join(person_dir, base_name)
                cv2.imwrite(dest, img)

            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
            img_idx += 1

        self.save_data(known_encodings, known_names)
        print(f"Added {len(images)} face encodings for {name}")
    
    def remove_person(self, name):
        """
        Removes all entries for a specified person from the system.

        Args:
            name (str): The name of the person to remove.

        This method deletes the person's data from the encodings file and removes their images.
        """
        known_encodings, known_names = self.load_existing_data()
        
        new_encodings = []
        new_names = []
        for enc, nm in zip(known_encodings, known_names):
            if nm != name:
                new_encodings.append(enc)
                new_names.append(nm)
        
        self.save_data(new_encodings, new_names)

        # Remove image folder
        image_dir = f"/home/alp/face_data/images/{name}"
        try:
            if os.path.exists(image_dir):
                # Remove all files in directory
                for filename in os.listdir(image_dir):
                    file_path = os.path.join(image_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                # Remove the directory itself
                os.rmdir(image_dir)
        except Exception as e:
            print(f"Error deleting image directory for {name}: {e}")
        
        print(f"Removed all data of {name}")

    def load_existing_data(self):
        """
        Loads existing face encodings and names from the data file.

        Returns:
            tuple: A tuple containing two lists: known_encodings and known_names.

        If the data file does not exist or is empty, it returns empty lists.
        """
        try:
            with open(self.data_file, 'rb') as f:
                known_encodings, known_names = pickle.load(f)
            return known_encodings, known_names
        except (FileNotFoundError, EOFError):
            return [], []
    
    def save_data(self, encodings, names):
        """
        Saves face encodings and names to the data file.

        Args:
            encodings (list): A list of face encodings to save.
            names (list): A list of names associated with the face encodings.

        This method also rebuilds the FAISS index every time the data changes.
        """
        with open(self.data_file, 'wb') as f:
            pickle.dump((encodings, names), f)
        # Rebuild FAISS index every time data changes
        self.save_faiss_index(encodings, names)
    
    def list_people(self):
        """
        Lists all people in the face recognition system.

        This method retrieves the names of all individuals and prints them along with 
        the count of images associated with each name.
        """
        _, known_names = self.load_existing_data()
        unique_names = sorted(set(known_names))
        print("\nPeople:")
        for name in unique_names:
            count = known_names.count(name)
            print(f"- {name} ({count} images)")
    
    def list_unknown_people(self):
        """
        Lists all unknown people detected in the system.

        This method checks the images directory for any unknown persons and prints their names.

        Returns:
            bool: True if unknown persons are found, False otherwise.
        """
        unknowns = []
        try:
            images_dir = "/home/alp/face_data/images"
            if os.path.exists(images_dir):
                unknowns = [d for d in os.listdir(images_dir) if d.startswith("unknown_person_")]
        except Exception as e:
            print(f"Error listing unknown persons: {e}")
        
        if not unknowns:
            print("No unknown person found")
            return False
        print("\nAvailable unknown persons:")
        for unknown in sorted(unknowns):
            print(f"- {unknown}")

        return True
        
    def rename_person(self, old_name, new_name):
        """
        Renames an existing person in the face recognition system.

        Args:
            old_name (str): The current name of the person to rename.
            new_name (str): The new name to assign to the person.

        Returns:
            bool: True if the renaming was successful, False otherwise.

        This method updates the name in the encodings file and renames the corresponding image directory.
        """
        if old_name == new_name:
            print("Old name and new name are the same.")
            return False

        old_dir = f"/home/alp/face_data/images/{old_name}"
        new_dir = f"/home/alp/face_data/images/{new_name}"
        
        if not os.path.exists(old_dir):
            print(f"Error: {old_name} does not exist")
            return False
        if os.path.exists(new_dir):
            print(f"Error: A person named '{new_name}' already exists")
            return False

        encodings, names = self.load_existing_data()
        updated = False
        for i in range(len(names)):
            if names[i] == old_name:
                names[i] = new_name
                updated = True
        
        if not updated:
            print(f"No entries found for {old_name}")
            return False

        self.save_data(encodings, names)
        os.rename(old_dir, new_dir)
        print(f"Renamed {old_name} to {new_name}")
        return True
    
    def interactive_menu(self):
        """
        Provides an interactive command line interface for managing face data.

        This method allows users to add, remove, rename, and list people in the face recognition system.
        """
        try:
            while True:
                print("\nFace Data Manager")
                print("1. Add person from camera")
                print("2. Add person from image files")
                print("3. Remove person")
                print("4. Remove unknown person")
                print("5. List all people")
                print("6. Define unknown person")
                print("7. Exit")
                
                choice = input("Select option: ")
                
                if choice == '1':
                    name = input("Enter person name: ")
                    num = input("Number of images to capture [3]: ")
                    self.capture_from_camera(name, int(num) if num.isdigit() else 3)
                
                elif choice == '2':
                    name = input("Enter person name: ")
                    dir_path = input("Enter directory with images: ")
                    if os.path.isdir(dir_path):
                        image_paths = [
                            os.path.join(dir_path, f) 
                            for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        ]
                        self.add_person(name, image_paths)
                    else:
                        print("Invalid directory path")
                
                elif choice == '3':
                    name = input("Enter name to remove: ")
                    self.remove_person(name)
                
                elif choice == '4':
                    # First list all unknown persons
                    unknowns = self.list_unknown_people()
                    
                    if not unknowns:
                        continue 

                    unknown_num = input("Enter unknown person number to remove: ")
                    if not unknown_num.isdigit():
                        print("Invalid number")
                        continue

                    self.remove_person(f"unknown_person_{unknown_num}")
                    
                elif choice == '5':
                    self.list_people()
                
                elif choice == '6':
                    # First list all unknown persons
                    unknowns = self.list_unknown_people()
                    
                    if not unknowns:
                        continue 
                    
                    unknown_num = input("Which unknown person to define, enter the number: ")
                    if not unknown_num.isdigit():
                        print("Invalid number")
                        continue
                    
                    new_name = input("Enter new name: ")
                    if not new_name:
                        print("Name cannot be empty")
                        continue
                    
                    self.rename_person(f"unknown_person_{unknown_num}", new_name)
                
                elif choice == '7':
                    break
        finally:
            self.capture.stop()

if __name__ == "__main__":
    """
    Main function for running the FaceDataManager.

    This function creates an instance of the FaceDataManager and starts the interactive menu.
    """
    manager = FaceDataManager()
    manager.interactive_menu()
