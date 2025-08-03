import os
import cv2
import threading
import socket
import json
import face_recognition
import numpy
import pickle
import faiss
import time
from collections import deque

class FaceRecognizer:
    """
    FaceRecognizer class for face recognition using FAISS.

    This class provides methods for loading face encodings, recognizing faces, 
    and rebuilding the FAISS index. It also includes helper methods for checking 
    if a face is frontal and for saving images.
    """
    def __init__(self,
                 index_file='/home/alp/face_data/faiss.index',
                 names_file='/home/alp/face_data/faiss_names.pkl',
                 data_file='/home/alp/face_data/encodings.dat'):
        """
        Initializes the FaceRecognizer with paths for the FAISS index, names, and encodings data.

        Args:
            index_file (str): Path to the FAISS index file.
            names_file (str): Path to the file containing names associated with face encodings.
            data_file (str): Path to the file containing face encodings.
        """
        self.index_file = index_file
        self.names_file = names_file
        self.data_file = data_file

        self.face_index = None  # FAISS index
        self.face_names = []
        self.face_encodings = []

        self.load_faces()

        self.unknown_person_count = 0
        # Get the last unknown person count if directories exist
        if os.path.exists("/home/alp/face_data"):
            existing_unknowns = [d for d in os.listdir("/home/alp/face_data/images") if d.startswith("unknown_person_")]
            if existing_unknowns:
                last_numbers = [int(d.split("_")[-1]) for d in existing_unknowns if d.split("_")[-1].isdigit()]
                if last_numbers:
                    self.unknown_person_count = max(last_numbers)
    
    def load_faces(self):
        """
        Loads face encodings and names from the specified data file and FAISS index.

        This method attempts to read the face encodings and names from the data file. 
        If the FAISS index file exists, it loads the index; otherwise, it prepares to rebuild it on the next add.

        Raises:
            Exception: If there is an error loading the face data.
        """
        try:
            # Load encodings and names from a single tuple
            with open(self.data_file, 'rb') as f:
                self.face_encodings, self.face_names = pickle.load(f)

            print(f"Loaded {len(self.face_names)} face names and {len(self.face_encodings)} encodings")

            # Load FAISS index
            if os.path.exists(self.index_file):
                self.face_index = faiss.read_index(self.index_file)
                print("Loaded FAISS index for faces")
            else:
                print("FAISS index file not found. Will rebuild on next add.")
                self.face_index = None

        except Exception as e:
            print(f"Could not load face data: {str(e)}")
            self.face_names = []
            self.face_encodings = []
            self.face_index = None

    def recognize_faces(self, frame):
        """
        Recognizes faces in a given frame using FAISS.

        Args:
            frame (numpy.ndarray): The image frame in which to recognize faces.

        Returns:
            list: A list of tuples containing recognized face names and their similarity scores.
                  Each tuple is in the format (name, similarity).

        This method processes the frame to find face locations and encodings, 
        then uses the FAISS index to identify faces and returns the results.
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
            
            # Early return if no faces or no known faces loaded
            if not face_encodings:
                cv2.imwrite(f"/home/alp/face_images/image_{x}.jpg", frame)
                return []
            
            if self.face_index is None or len(self.face_names) == 0:
                cv2.imwrite(f"/home/alp/face_images/image_{x}.jpg", frame)
                return [("Unknown", 0)] * len(face_encodings)

            results = []
            valid_face_indices = []  # Store indices of faces that are frontal

            for i, landmarks in enumerate(face_landmarks_list):
                # Check if face is frontal using landmarks
                if self.is_face_frontal(landmarks):
                    valid_face_indices.append(i)
            
            if not valid_face_indices:
                cv2.imwrite(f"/home/alp/face_images/image_{x}.jpg", frame)
                return []
            
            # Only process valid frontal faces
            valid_face_encodings = [face_encodings[i] for i in valid_face_indices]
            valid_face_locations = [face_locations[i] for i in valid_face_indices]
            
            face_vectors = numpy.array(valid_face_encodings).astype('float32')
            
            # Batch process known faces with FAISS
            distances, indices = self.face_index.search(face_vectors, 1)

            for i, (distance, index) in enumerate(zip(distances, indices)):                
                # Process known faces
                similarity = max(0, 1 - distance[0]) * 100
                similarity = round(similarity, 2)  # Round here for consistency
                
                if similarity >= 60.0:
                    results.append((self.face_names[index[0]], similarity))
                    continue
                        
                # If we get here, it's a new unknown face
                results.append(("new_unknown", similarity))
            
            # Handle unknown person tracking if exactly one new unknown face is detected
            if len(results) == 1 and results[0][0] == "new_unknown" and len(valid_face_locations) == 1:
                # Create new unknown person directory
                self.unknown_person_count += 1
                unknown_name = f"unknown_person_{self.unknown_person_count}"
                unknown_dir = f"/home/alp/face_data/images/{unknown_name}"
                os.makedirs(unknown_dir, exist_ok=True)
                
                # Save the image
                img_path = f"{unknown_dir}/0.jpg"
                cv2.imwrite(img_path, frame)
                
                # Save the encoding to encodings.dat
                self.face_encodings.append(valid_face_encodings[0])
                self.face_names.append(unknown_name)
                
                # Update the encodings.dat file
                with open(self.data_file, 'wb') as f:
                    pickle.dump((self.face_encodings, self.face_names), f)

                self.rebuild_faiss_index()
                
                print(f"Created new {unknown_name}, saved image and updated encodings.dat & faiss index")
            
            # If recognized as existing unknown person, add more images if needed
            elif len(results) == 1 and results[0][0].startswith("unknown_person_") and len(valid_face_locations) == 1:
                unknown_name = results[0][0]
                # Check how many images we already have
                images_dir = f"/home/alp/face_data/images/{unknown_name}"
                
                if os.path.exists(images_dir):
                    existing_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
                    
                    # Only save if we have fewer than 5 images
                    if len(existing_images) < 5:
                        img_path = f"{images_dir}/{len(existing_images)}.jpg"
                        cv2.imwrite(img_path, frame)
                        
                        # Add this encoding to encodings.dat as well
                        self.face_encodings.append(valid_face_encodings[0])
                        self.face_names.append(unknown_name)
                        
                        # Update the encodings.dat file
                        with open(self.data_file, 'wb') as f:
                            pickle.dump((self.face_encodings, self.face_names), f)
                        
                        self.rebuild_faiss_index()

                        print(f"Added image {len(existing_images)+1}/5 for {unknown_name} and updated encodings.dat & faiss index")

            # Draw rectangles and labels on the frame
            for (name, similarity), (top, right, bottom, left) in zip(results, valid_face_locations):
                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                # Prepare the label with name and similarity
                label = f"{name}: {similarity:.2f}%"
                # Draw the label above the rectangle
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # WRITE MODIFIED FRAME
            cv2.imwrite(f"/home/alp/face_images/image_{x}.jpg", frame)

            return results
        except Exception as e:
            print(f"Face recognition error: {str(e)}")
            return []
        
    def rebuild_faiss_index(self):
        """
        Rebuilds the FAISS index using the current face encodings.

        This method creates a new FAISS index and adds the current face encodings to it. 
        It also saves the updated index to the specified index file and the encodings 
        and names to the data file.

        Raises:
            Exception: If there is an error rebuilding the FAISS index.
        """
        try:
            encoding_matrix = numpy.array(self.face_encodings).astype('float32')
            self.face_index = faiss.IndexFlatL2(encoding_matrix.shape[1])
            self.face_index.add(encoding_matrix)
            faiss.write_index(self.face_index, self.index_file)
            
            # Save both encodings and names together
            with open(self.data_file, 'wb') as f:
                pickle.dump((self.face_encodings, self.face_names), f)

            print("FAISS index rebuilt and face data saved.")
        except Exception as e:
            print(f"Failed to rebuild FAISS index: {e}")

    def is_face_frontal(self, landmarks):
        """
        Checks if a face is frontal based on facial landmarks.

        Args:
            landmarks (dict): A dictionary containing facial landmarks.

        Returns:
            bool: True if the face is frontal, False if it's at a significant angle.

        This method analyzes the positions of key facial landmarks to determine 
        if the face is oriented towards the camera.
        """
        # Get the important landmarks
        nose_bridge = numpy.array(landmarks['nose_bridge'])
        chin = numpy.array(landmarks['chin'])
        left_eye = numpy.array(landmarks['left_eye'])
        right_eye = numpy.array(landmarks['right_eye'])
        
        # Calculate eye centers
        left_eye_center = numpy.mean(left_eye, axis=0)
        right_eye_center = numpy.mean(right_eye, axis=0)
        
        # Calculate eye distance
        eye_distance = numpy.linalg.norm(left_eye_center - right_eye_center)
        
        # Calculate nose direction (from top to bottom of nose bridge)
        nose_direction = nose_bridge[-1] - nose_bridge[0]
        
        # Calculate chin position relative to nose
        chin_position = chin[8]  # Middle of chin
        
        # For a frontal face:
        # 1. The nose direction should be mostly vertical
        # 2. The chin should be centered below the nose
        # 3. The eyes should be roughly horizontal
        
        # Check nose verticality (x component should be small compared to y component)
        if abs(nose_direction[0]) > 0.3 * abs(nose_direction[1]):
            return False
        
        # Check chin position (should be roughly centered below nose)
        nose_bottom = nose_bridge[-1]
        if abs(chin_position[0] - nose_bottom[0]) > 0.3 * eye_distance:
            return False
        
        # Check eye alignment (should be roughly horizontal)
        if abs(left_eye_center[1] - right_eye_center[1]) > 0.2 * eye_distance:
            return False
        
        return True


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


class NetworkSender:
    """
    NetworkSender class for sending data over a network connection.

    This class provides methods for connecting to a server, sending data, 
    and managing the connection state.
    """
    def __init__(self, host, port):
        """
        Initializes the NetworkSender with the specified host and port.

        Args:
            host (str): The hostname or IP address of the server to connect to.
            port (int): The port number of the server to connect to.
        """
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.lock = threading.Lock()
        
    def connect(self):
        """
        Connects to the remote server.

        This method attempts to establish a connection to the specified host and port. 
        If successful, it sets the connected flag to True.
        """
        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            
    def send_data(self, data):
        """
        Sends data over the network to the connected server.

        Args:
            data (dict): The data to send, which will be converted to JSON.

        Returns:
            bool: True if the data was sent successfully, False otherwise.

        This method ensures that the connection is established before attempting to send data. 
        If the connection fails, it tries to reconnect.
        """
        if not self.connected:
            self.connect()
            if not self.connected:
                return False
                
        try:
            with self.lock:
                # Convert data to JSON and send
                json_data = json.dumps(data)
                self.socket.sendall(json_data.encode('utf-8') + b'\n')
                return True
        except Exception as e:
            print(f"Error sending data: {e}")
            self.connected = False
            return False
            
    def close(self):
        """
        Closes the connection to the server.

        This method attempts to close the socket connection and sets the connected flag to False.
        """
        try:
            self.socket.close()
            self.connected = False
        except:
            pass


if __name__ == "__main__":
    """
    Main entry point for the face recognition system.

    This script initializes the face recognizer, network sender, frame capture, 
    and processing time tracker, then continuously captures frames, processes them, 
    and sends the results to the network.
    """
    # Define paths
    images_dir = "/home/alp/face_images"
    
    # Initialize directories
    os.system(f"rm -rf {images_dir}")
    os.makedirs(images_dir, exist_ok=True)

    face_recognizer = FaceRecognizer()

    # Initialize network sender (with IP and port)
    network_sender = NetworkSender("127.0.0.1", 9998)

    # Initialize frame capture
    frame_capture = FrameCapture("rtsp://admin:gsmtest123@10.11.101.100:554/Streaming/channels/101")
    frame_capture.start()

    # Initialize processing time tracker
    processing_times = deque(maxlen=10)  # Stores last 10 processing times

    x = 0
    try:
        while True:
            start_time = time.time()
            
            # Get the most recent frame
            frame = frame_capture.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue
                
            # Process the frame
            face_results = face_recognizer.recognize_faces(frame)
            face_results_rounded = [(name, round(float(similarity), 2)) for name, similarity in face_results]
            print(f"Frame {x}: {face_results_rounded}")

            # Prepare data to send
            data_to_send = {
                "frame_id": x,
                "timestamp": time.time(),
                "face_recognition": face_results_rounded,
                "processing_time": time.time() - start_time
            }

            # Send data over network
            network_sender.send_data(data_to_send)
            
            x += 1
            
            # Calculate and track processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Calculate average of last 10 processing times
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Print timing information
            print(f"Current processing time: {processing_time:.2f}s")
            print(f"Average of last 10: {avg_processing_time:.2f}s\n")
    
    except KeyboardInterrupt:
        frame_capture.stop()
        network_sender.close()
        