import socket
import json
import threading
from queue import Queue
from time import sleep

class TriplePortReceiver:
    """
    TriplePortReceiver class for receiving data from three separate services.

    This class provides methods for starting and stopping the receiver, 
    handling incoming connections from each service, and processing the data streams.
    """
    def __init__(self, face_port=9998, security_port=9999, speech_port=10000):
        """
        Initializes the TriplePortReceiver with specified ports for face recognition,
        security analysis, and speech processing services.

        Args:
            face_port (int): Port for the face recognition service.
            security_port (int): Port for the security analysis service.
            speech_port (int): Port for the speech service.
        """
        # Port configuration
        self.face_port = face_port
        self.security_port = security_port
        self.speech_port = speech_port
        
        # Separate queues for each stream
        self.face_queue = Queue()
        self.security_queue = Queue()
        self.speech_queue = Queue()
        
        # Control flag
        self.running = False
        
        # Locks for thread safety
        self.face_lock = threading.Lock()
        self.security_lock = threading.Lock()
        self.speech_lock = threading.Lock()

    def handle_face_connection(self, conn):
        """
        Handles incoming connections for the face recognition service.

        Args:
            conn (socket.socket): The socket connection object for the client.

        This method listens for data from the face recognition client, 
        decodes the JSON data, and places valid messages into the face queue.
        If the connection is lost or invalid data is received, it handles the exceptions accordingly.
        """
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    if 'face_recognition' in message:  # Validate face recognition message
                        with self.face_lock:
                            self.face_queue.put(message)
                except json.JSONDecodeError:
                    print("Invalid JSON received on face recognition port")
        except ConnectionResetError:
            print("Face recognition client disconnected")
        finally:
            conn.close()

    def handle_security_connection(self, conn):
        """
        Handles incoming connections for the security analysis service.

        Args:
            conn (socket.socket): The socket connection object for the client.

        This method listens for data from the security analysis client, 
        decodes the JSON data, and places valid messages into the security queue.
        If the connection is lost or invalid data is received, it handles the exceptions accordingly.
        """
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    if 'threat_level' in message:  # Validate security message
                        with self.security_lock:
                            self.security_queue.put(message)
                except json.JSONDecodeError:
                    print("Invalid JSON received on security port")
        except ConnectionResetError:
            print("Security client disconnected")
        finally:
            conn.close()

    def handle_speech_connection(self, conn):
        """
        Handles incoming connections for the speech service.

        Args:
            conn (socket.socket): The socket connection object for the client.

        This method listens for data from the speech client, 
        decodes the JSON data, and places valid messages into the speech queue.
        If the connection is lost or invalid data is received, it handles the exceptions accordingly.
        """
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    if 'speech' in message:  # Validate speech message
                        with self.speech_lock:
                            self.speech_queue.put(message)
                except json.JSONDecodeError:
                    print("Invalid JSON received on speech port")
        except ConnectionResetError:
            print("Speech client disconnected")
        finally:
            conn.close()

    def start_face_server(self):
        """
        Starts the face recognition server to listen for incoming connections.

        This method binds the server to the specified face recognition port and 
        starts listening for client connections. Each connection is handled in a 
        separate thread using the handle_face_connection method.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', self.face_port))
            s.listen()
            print(f"Face recognition service listening on port {self.face_port}")
            
            while self.running:
                conn, addr = s.accept()
                print(f"Face recognition service connected by {addr}")
                threading.Thread(
                    target=self.handle_face_connection,
                    args=(conn,),
                    daemon=True
                ).start()

    def start_security_server(self):
        """
        Starts the security analysis server to listen for incoming connections.

        This method binds the server to the specified security analysis port and 
        starts listening for client connections. Each connection is handled in a 
        separate thread using the handle_security_connection method.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', self.security_port))
            s.listen()
            print(f"Security analysis service listening on port {self.security_port}")
            
            while self.running:
                conn, addr = s.accept()
                print(f"Security analysis service connected by {addr}")
                threading.Thread(
                    target=self.handle_security_connection,
                    args=(conn,),
                    daemon=True
                ).start()

    def start_speech_server(self):
        """
        Starts the speech service server to listen for incoming connections.

        This method binds the server to the specified speech service port and 
        starts listening for client connections. Each connection is handled in a 
        separate thread using the handle_speech_connection method.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', self.speech_port))
            s.listen()
            print(f"Speech service listening on port {self.speech_port}")
            
            while self.running:
                conn, addr = s.accept()
                print(f"Speech service connected by {addr}")
                threading.Thread(
                    target=self.handle_speech_connection,
                    args=(conn,),
                    daemon=True
                ).start()

    def process_face_data(self, data):
        """
        Processes face recognition data received from the face service.

        Args:
            data (dict): The data received from the face recognition service, 
                          expected to contain frame ID, timestamp, and face recognition results.

        This method prints the frame ID, timestamp, and details of detected faces 
        including their names and similarity scores.
        #TODO: Implement face recognition logic here
        """
        print("\n=== Face Recognition Data ===")
        print(f"Frame ID: {data.get('frame_id', 'N/A')}")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        print("Faces detected:")
        for name, similarity in data.get('face_recognition', []):
            print(f"- {name}: {similarity:.2f}%")
        print("="*30 + "\n")

    def process_security_data(self, data):
        """
        Processes security analysis data received from the security service.

        Args:
            data (dict): The data received from the security analysis service, 
                          expected to contain frame ID, timestamp, roles, smoking detection, 
                          cars detection, and threat level.

        This method prints the frame ID, timestamp, roles detected, and other 
        security-related information.
        #TODO: Implement security analysis logic here
        """
        print("\n=== Security Analysis Data ===")
        print(f"Frame ID: {data.get('frame_id', 'N/A')}")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"Roles: {data.get('roles', [])}")
        print(f"Smoking detected: {data.get('smoking', False)}")
        print(f"Cars detected: {data.get('cars', False)}")
        print(f"Threat Level: {data.get('threat_level', 0)}")
        print("="*30 + "\n")

    def process_speech_data(self, data):
        """
        Processes speech data received from the speech service.

        Args:
            data (dict): The data received from the speech service, expected to contain speech text.

        This method prints the speech data received from the speech service.
        #TODO: Implement speech analysis logic here
        """
        print("\n=== Speech Data ===")
        print(f"Speech: {data.get('speech', '')}")
        print("="*30 + "\n")

    def start(self):
        """
        Starts all servers and processing loops for the services.

        This method initializes the running state, starts the face, security, 
        and speech servers in separate threads, and begins processing the data 
        streams from each service.
        """
        self.running = True
        
        # Start all servers in separate threads
        threading.Thread(target=self.start_face_server, daemon=True).start()
        threading.Thread(target=self.start_security_server, daemon=True).start()
        threading.Thread(target=self.start_speech_server, daemon=True).start()
        
        # Start processing threads for each stream
        threading.Thread(target=self.process_face_stream, daemon=True).start()
        threading.Thread(target=self.process_security_stream, daemon=True).start()
        threading.Thread(target=self.process_speech_stream, daemon=True).start()
        
        try:
            while self.running:
                sleep(1)  # Keep main thread alive
        except KeyboardInterrupt:
            self.stop()

    def process_face_stream(self):
        """
        Processes the face recognition stream from the face queue.

        This method continuously checks the face queue for new data and 
        processes it using the process_face_data method.
        """
        while self.running:
            if not self.face_queue.empty():
                data = self.face_queue.get()
                self.process_face_data(data)
            sleep(0.01)  # Small sleep to prevent busy waiting

    def process_security_stream(self):
        """
        Processes the security analysis stream from the security queue.

        This method continuously checks the security queue for new data and 
        processes it using the process_security_data method.
        """
        while self.running:
            if not self.security_queue.empty():
                data = self.security_queue.get()
                self.process_security_data(data)
            sleep(0.01)

    def process_speech_stream(self):
        """
        Processes the speech stream from the speech queue.

        This method continuously checks the speech queue for new data and 
        processes it using the process_speech_data method.
        """
        while self.running:
            if not self.speech_queue.empty():
                data = self.speech_queue.get()
                self.process_speech_data(data)
            sleep(0.01)

    def stop(self):
        """
        Cleans up and shuts down the server.

        This method sets the running flag to False, allowing all threads to exit 
        gracefully and prints a shutdown message.
        """
        self.running = False
        print("Receiver shutting down...")

if __name__ == "__main__":
    """
    Main function for running the TriplePortReceiver.

    This function creates an instance of the TriplePortReceiver and starts it, 
    allowing it to receive data from the face recognition, security analysis, 
    and speech services.
    """
    receiver = TriplePortReceiver(
        face_port=9998,      # Port for face recognition service
        security_port=9999,  # Port for security analysis service
        speech_port=10000    # Port for speech service
    )
    receiver.start()    # Start the receiver
    