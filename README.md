# Federated Learning System For Primary Healthcare Centres (PHCs) with Differential Privacy

A complete federated learning implementation using TensorFlow.js for client-side training, with MobileNetV2 for image classification tasks. The system includes differential privacy mechanisms and secure weight aggregation across multiple clients.

##  Overview

This federated learning system enables multiple clients to collaboratively train a machine learning model without sharing raw data. The architecture consists of:

- **Frontend Client**: Web-based interface for local model training
- **Edge Server**: Intermediate aggregation point for client weights
- **Central Server**: Final federated averaging and global model distribution

##  Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client 1  │    │   Client 2  │    │   Client N  │
│  (Frontend) │    │  (Frontend) │    │  (Frontend) │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                  ┌─────────────┐
                  │ Edge Server │
                  │  (Flask)    │
                  └─────────────┘
                           │
                  ┌─────────────┐
                  │   Central   │
                  │   Server    │
                  │  (Flask)    │
                  └─────────────┘
```

##  Features

### Client-Side Features
- **Local Training**: Train MobileNetV2 model on local image data
- **Differential Privacy**: Adds minimal noise to weights before sharing
- **File Upload**: Drag-and-drop interface for dataset upload
- **Real-time Monitoring**: Training progress and accuracy tracking
- **Validation**: Local validation of global model

### Server-Side Features
- **Federated Averaging**: Weighted aggregation of client models
- **AES Encryption**: Secure communication between servers
- **Automatic Triggering**: Starts aggregation when minimum clients reached
- **Health Monitoring**: Status endpoints for system health
- **Weight Validation**: Ensures proper weight structure

##  Prerequisites

- **Python 3.8+** for server components
- **Modern web browser** with WebGL support
- **Node.js** (optional, for serving frontend)

### Python Dependencies
```bash
pip install flask flask-cors pycryptodome numpy requests
```

##  Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd federated-learning-system
```

### 2. Server Setup

#### Central Server
```bash
cd server
python serverapp.py
```
Default port: `8001`

#### Edge Server
```bash
cd edge-server
python edgeserverapp.py
```
Default port: `8000`

### 3. Frontend Setup
```bash
cd frontend
# Serve using any HTTP server
python -m http.server 3000
# or
npx serve .
```

### 4. Environment Configuration

Create `.env` files for server configuration:

**Central Server (.env)**
```env
PORT=8001
AES_KEY=000102030405060708090a0b0c0d0e0f
AES_IV=101112131415161718191a1b1c1d1e1f
EDGE_SERVER_URL=http://localhost:8000
```

**Edge Server (.env)**
```env
PORT=8000
CENTRAL_SERVER_URL=http://localhost:8001
SERVER_AES_KEY=000102030405060708090a0b0c0d0e0f
SERVER_AES_IV=101112131415161718191a1b1c1d1e1f
```

##  Dataset Format

The system expects image files with specific naming convention:

```
{image_number}_{label}.{extension}
```

### Examples:
- `0_1.jpg` - Image 0 with label 1
- `5_0.png` - Image 5 with label 0
- `10_1.jpeg` - Image 10 with label 1

### Supported:
- **Image formats**: JPG, PNG, JPEG
- **Labels**: Binary classification (0 or 1)
- **Image size**: Automatically resized to 96x96 pixels

##  Configuration

### Client Configuration (`app.js`)
```javascript
const CONFIG = {
    EDGE_SERVER_URL: 'http://localhost:8000',
    IMAGE_SIZE: 96,
    BATCH_SIZE: 8,
    EPOCHS: 7,
    LEARNING_RATE: 5e-5,
};
```

### Training Parameters
- **Epochs**: 7 (adjustable)
- **Batch Size**: 8
- **Learning Rate**: 5e-5 with exponential decay
- **Validation Split**: 20%
- **Model**: MobileNetV2 (last 42 layers trainable)



##  Usage

### Step 1: Start Servers
1. Start Central Server: `python serverapp.py`
2. Start Edge Server: `python edgeserverapp.py`
3. Verify connectivity at health endpoints

### Step 2: Prepare Data
1. Organize images with proper naming convention
2. Ensure balanced dataset for better performance
3. Recommended: 20+ images per class

### Step 3: Train Clients
1. Open frontend in multiple browser tabs/windows
2. Upload different datasets to each client
3. Click "Start Local Training"
4. Monitor training progress

### Step 4: Federated Learning
1. System automatically triggers aggregation when 2+ clients complete training
2. Global model is computed using federated averaging
3. Clients receive and validate the global model
4. View final validation accuracy

##  Security Features

### Encryption
- **AES-256-CBC**: Secure server-to-server communication
- **Base64 Encoding**: Safe data transmission
- **Environment Variables**: Secure key management

### Privacy Protection
- **Differential Privacy**: Adds calibrated noise to weights
- **Local Processing**: Raw data never leaves client device
- **Weight-Only Sharing**: Only model parameters are transmitted

##  Monitoring & Debugging

### Health Endpoints
- **Central Server**: `GET /health`
- **Edge Server**: `GET /health`
- **Status Check**: `GET /status`

### Logging
- Real-time logs in web interface
- Server-side Python logging
- Training metrics and validation results

### Common Issues
1. **Connection Refused**: Check server ports and URLs
2. **Invalid File Format**: Ensure proper image naming
3. **Training Fails**: Verify dataset size and balance
4. **Encryption Errors**: Check AES key configuration

##  API Endpoints

### Edge Server
- `POST /uploadWeights` - Receive client weights
- `GET /getGlobalModel` - Provide global model to clients
- `POST /receiveGlobalModel` - Receive from central server
- `GET /status` - Server status and connectivity

### Central Server
- `POST /uploadWeights` - Receive aggregated weights
- `GET /globalWeights` - Get current global model
- `GET /health` - Health check
- `POST /reset` - Reset server state

## Testing

### Unit Testing
```bash
# Test server endpoints
curl -X GET http://localhost:8001/health
curl -X GET http://localhost:8000/health
```

### Integration Testing
1. Start all servers
2. Upload test datasets to multiple clients
3. Verify successful aggregation
4. Check global model distribution

## Development

### Frontend Development
- **Framework**: Vanilla JavaScript with TensorFlow.js
- **Model**: Pre-trained MobileNetV2
- **UI**: Responsive design with progress tracking

### Backend Development
- **Framework**: Flask with CORS support
- **Encryption**: PyCryptodome for AES
- **Validation**: Comprehensive weight structure validation

### Adding New Features
1. **New Model Architecture**: Modify `createModel()` function
2. **Different Privacy Mechanisms**: Update `addMinimalNoise()`
3. **Additional Metrics**: Extend logging and monitoring

## Model Details

### MobileNetV2 Configuration
- **Input Size**: 96x96x3 RGB images
- **Output**: Binary classification (sigmoid activation)
- **Trainable Layers**: Last 42 layers unfrozen
- **Optimizer**: Adam with exponential decay
- **Loss Function**: Binary crossentropy

### Training Process
1. **Data Preprocessing**: Images resized and normalized
2. **Local Training**: 7 epochs with validation split
3. **Weight Extraction**: Systematic weight serialization
4. **Noise Addition**: Differential privacy application
5. **Aggregation**: Federated averaging algorithm

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature`)
3. Commit changes (`git commit -m 'Add  feature'`)
4. Push to branch (`git push origin feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**Note**: This is a research/educational implementation. For production use, additional security measures and optimizations should be implemented.
