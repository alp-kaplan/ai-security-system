# AI Security System

A comprehensive AI-powered security monitoring system that combines face recognition, threat detection, and security analysis using computer vision and large language models.

## 🌟 Features

### 🎯 Face Recognition System
- **Real-time face detection and recognition** using FAISS for fast similarity search
- **Automatic unknown person tracking** - creates profiles for unrecognized individuals
- **Frontal face validation** - ensures quality face captures using landmark analysis
- **Dynamic face database management** with automatic FAISS index rebuilding
- **Web-based face management interface** for adding, removing, and renaming people

### 🔒 Security Analysis
- **AI-powered threat detection** using Qwen2.5-VL-7B vision language model
- **Role classification** - identifies workers, security personnel, delivery staff, and strangers
- **Smoking detection** in monitored areas
- **Vehicle presence detection**
- **Threat level assessment** (0-10 scale)

### 🏢 System Coordination
- **Multi-service architecture** with dedicated network communication
- **Real-time data streaming** between services
- **Centralized monitoring** through the coordinator service

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Face           │    │  Security       │    │  Speech         │
│  Recognition    │    │  Analysis       │    │  Processing     │
│  (Port 9998)    │    │  (Port 9999)    │    │  (Port 10000)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │     Coordinator         │
                    │     (kapici.py)         │
                    └─────────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │   Face Manager Web UI   │
                    │   (Port 8000)           │
                    └─────────────────────────┘
```

## 📦 Components

### 1. Face Recognition (`face_recognizer.py`)
- **FaceRecognizer**: Main face recognition engine with FAISS integration
- **FrameCapture**: RTSP video stream capture with threading
- **NetworkSender**: Communicates results to coordinator

### 2. Security Analysis (`security_qwen7.py`)
- **Security**: AI model for security threat analysis
- **Role Classification**: Identifies person types in surveillance area
- **Threat Assessment**: Multi-factor security evaluation

### 3. Coordinator (`kapici.py`)
- **TriplePortReceiver**: Manages multiple service connections
- **Data Processing**: Handles incoming streams from all services
- **Event Coordination**: Centralizes security event management

### 4. Face Manager (`face_manager/`)
- **Web Interface**: Browser-based face data management
- **REST API**: Backend for face operations
- **Interactive CLI**: Command-line face management tool

## 🚀 Installation

### Prerequisites
```bash
# System dependencies
sudo apt update
sudo apt install python3-pip python3-dev cmake
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `face-recognition` - Face detection and encoding
- `faiss-cpu` - Fast similarity search
- `opencv-python` - Computer vision operations
- `transformers` - Qwen2.5-VL model support
- `torch` - Deep learning framework
- `fastapi` - Web API framework
- `uvicorn` - ASGI server

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for Qwen2.5-VL model
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 10GB+ for models and face data

## ⚙️ Configuration

### RTSP Stream Setup
Update the RTSP URL in the configuration files:
```python
# Current configuration
rtsp_url = "rtsp://admin:gsmtest123@10.11.101.100:554/Streaming/channels/101"
```

### Directory Structure
The system creates the following directory structure:
```
/home/alp/face_data/
├── faiss.index              # FAISS search index
├── faiss_names.pkl          # Associated names
├── encodings.dat            # Face encodings data
└── images/                  # Face image storage
    ├── person_name/         # Individual person folders
    └── unknown_person_X/    # Auto-detected unknown persons
```

## 🎮 Usage

### Starting the System

#### 1. Launch Face Recognition Service
```bash
python face_recognizer.py
```

#### 2. Start Security Analysis Service
```bash
python security_qwen7.py
```

#### 3. Run Coordinator Service
```bash
python kapici.py
```

#### 4. Launch Face Manager Web Interface
```bash
cd face_manager/
uvicorn backend:app --host 0.0.0.0 --port 8000
```

#### 5. Access Face Manager CLI (Optional)
```bash
cd face_manager/
python face_manager.py
```

### Web Interface
Navigate to `http://localhost:8000` to access the face management interface where you can:
- Add new people with photos
- Remove existing persons
- Rename individuals
- View all registered faces

### Command Line Interface
The CLI provides interactive options for:
1. Add person from camera
2. Add person from image files
3. Remove person
4. Remove unknown person
5. List all people
6. Define unknown person
7. Exit

## 📊 Data Flow

1. **Video Capture**: RTSP streams are captured in real-time
2. **Face Detection**: Faces are detected and encoded using face_recognition
3. **Face Matching**: FAISS performs similarity search against known faces
4. **Security Analysis**: Qwen2.5-VL analyzes frames for threats and roles
5. **Data Transmission**: Results are sent via TCP sockets to coordinator
6. **Event Processing**: Coordinator processes and logs security events

## 🔧 API Endpoints

### Face Manager API
- `GET /` - Serve web interface
- `GET /list` - List all registered people
- `GET /list-unknown` - List unknown persons
- `POST /add` - Add new person with images
- `DELETE /delete/{name}` - Remove person
- `POST /rename` - Rename person
- `GET /people_with_images` - Get people with image previews

## 🛡️ Security Features

### Face Recognition Security
- **Frontal face validation** prevents profile/angled face false positives
- **Similarity thresholding** (60%+ for positive matches)
- **Unknown person tracking** with automatic profile creation
- **FAISS indexing** for fast, scalable face search

### AI Security Analysis
- **Multi-modal analysis** combining visual and contextual information
- **Role-based access control** identification
- **Behavioral pattern detection** (smoking, loitering)
- **Vehicle monitoring** in restricted areas
- **Graduated threat assessment** with numerical scoring

## 🐛 Troubleshooting

### Common Issues

**CUDA/GPU Issues:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**RTSP Connection Issues:**
- Verify camera IP and credentials
- Check network connectivity
- Ensure camera supports the specified RTSP path

**Face Recognition Issues:**
- Ensure adequate lighting for face detection
- Verify camera angle captures frontal faces
- Check `/home/alp/face_data/` directory permissions

**Model Loading Issues:**
- Verify internet connection for model downloads
- Check available disk space (models are several GB)
- Ensure sufficient RAM for model loading

## 📝 Logging and Monitoring

The system provides comprehensive logging:
- **Processing times** for performance monitoring
- **Recognition results** with confidence scores
- **Security events** with threat levels
- **Network communication** status

## 🙏 Acknowledgments

- **face_recognition** library by Adam Geitgey
- **FAISS** by Facebook AI Research
- **Qwen2.5-VL** by Alibaba Cloud
- **OpenCV** community for computer vision tools
