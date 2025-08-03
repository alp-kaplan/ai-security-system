from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import re
import os
import cv2
import threading
import time
from collections import deque
import socket
import json

class Security:
    """
    Security class for analyzing video frames for threats, smoking, and vehicle presence.

    This class provides methods for loading a model, processing images, and parsing the model's output.
    """
    def __init__(self):
        """
        Initializes the Security class by loading the model and processor.

        This method sets up the model and processor for analyzing video frames.
        """
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).to("cuda")
        
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            min_pixels=min_pixels, 
            max_pixels=max_pixels, 
            use_fast=True
        )

    def parse(self, output_text):
        """
        Parses the model output text and returns structured information.
        Handles two formats:
        1. Proper format: 'roles: role1, role2,... or none|smoking: yes/no|cars: yes/no|threat level: number'
        2. Simplified format: 'none|no|yes|3' (in order: roles, smoking, cars, threat)
        
        Args:
            output_text (str): The raw output text from the model
            
        Returns:
            list: A list in the format [[role1, role2, ...], smoking?, cars?, threat_level]
                where smoking and cars are booleans, and threat_level is an integer
        """
        roles = []
        smoking = False
        cars = False
        threat_level = 0
        
        if '|' in output_text and not any(x in output_text.lower() for x in ['roles:', 'smoking:', 'cars:', 'threat level:']):
            parts = output_text.split('|')
            if len(parts) >= 4:
                roles_str = parts[0].strip()
                if roles_str.lower() != 'none':
                    roles = [role.strip() for role in roles_str.split(',')]
                
                smoking = parts[1].strip().lower() == 'yes'
                
                cars = parts[2].strip().lower() == 'yes'
                
                try:
                    threat_level = int(parts[3].strip())
                except ValueError:
                    threat_level = 0

        else:
            roles_match = re.search(r'roles:\s*(.*?)(?:\s*\||$)', output_text, re.IGNORECASE)
            if roles_match:
                roles_str = roles_match.group(1).strip()
                if roles_str.lower() != 'none':
                    roles = [role.strip() for role in roles_str.split(',')]
            
            smoking_match = re.search(r'smoking:\s*(yes|no)(?:\s*\||$)', output_text, re.IGNORECASE)
            if smoking_match:
                smoking = smoking_match.group(1).lower() == 'yes'
            
            cars_match = re.search(r'cars:\s*(yes|no)(?:\s*\||$)', output_text, re.IGNORECASE)
            if cars_match:
                cars = cars_match.group(1).lower() == 'yes'
            
            threat_match = re.search(r'threat level:\s*(\d+)(?:\s*\||$)', output_text, re.IGNORECASE)
            if threat_match:
                try:
                    threat_level = int(threat_match.group(1))
                except ValueError:
                    threat_level = 0
        
        return [roles, smoking, cars, threat_level]


    def infer(self, image):
        """
        Processes an image and returns classification results.

        Args:
            image: OpenCV image to be processed.

        Returns:
            list: Parsed output in format [[roles], smoking, cars, threat_level].

        This method sends the image to the model for inference and processes the output.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": "Classify the role of each person in the image: worker, security, delivery, stranger. Check for smoking, cars and threats. Output format: roles: role1, role2,... or none|smoking: yes/no|cars: yes/no|threat level: a number from 0 to 10. Do not output anything else."
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self.parse(output_text)
    

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
    Main function for running the security analysis service.

    This function initializes the security classifier, network sender, 
    and frame capture, and starts the frame capture thread. It then 
    continuously processes frames from the video stream, analyzes them, 
    and sends the results to the server.
    """
    # Define paths
    outputs_dir = "/home/alp/sec_outputs"
    
    # Initialize directories
    os.system(f"rm -rf {outputs_dir}")
    os.makedirs(outputs_dir, exist_ok=True)

    classifier = Security()

    # Initialize network sender (with IP and port)
    network_sender = NetworkSender("127.0.0.1", 9999)

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
            output = classifier.infer(frame)
            print(f"Frame {x}: {output}")

            # Prepare data to send
            data_to_send = {
                "frame_id": x,
                "timestamp": time.time(),
                "roles": output[0],
                "smoking": output[1],
                "cars": output[2],
                "threat_level": output[3],
                "processing_time": time.time() - start_time
            }

            # Send data over network
            network_sender.send_data(data_to_send)
            
            # Save results (for test purposes)
            with open(f"/home/alp/sec_outputs/output_{x}.txt", "w") as f:
                f.write(
                    f"{output[0]}\n"
                    f"Smoking: {output[1]}\n"
                    f"Cars: {output[2]}\n"
                    f"Threat Level: {output[3]}"
                )
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
