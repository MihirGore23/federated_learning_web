from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_model_storage = None

app = Flask(__name__)
CORS(app)

CENTRAL_SERVER_URL = os.getenv('CENTRAL_SERVER_URL', 'http://localhost:8001')
SERVER_AES_KEY = bytes.fromhex(os.getenv('SERVER_AES_KEY', '000102030405060708090a0b0c0d0e0f'))
SERVER_AES_IV = bytes.fromhex(os.getenv('SERVER_AES_IV', '101112131415161718191a1b1c1d1e1f'))

def decrypt_aes(encrypted_text):
    try:
        # Only for server communication
        encrypted_bytes = base64.b64decode(encrypted_text)
        cipher = AES.new(SERVER_AES_KEY, AES.MODE_CBC, SERVER_AES_IV)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        plaintext = unpad(decrypted_bytes, AES.block_size).decode("utf-8")
        return plaintext
    except Exception as e:
        logger.error(f"AES decryption failed: {str(e)}")
        raise
def encrypt_aes(plaintext):
    try:
        # Only for server communication
        cipher = AES.new(SERVER_AES_KEY, AES.MODE_CBC, SERVER_AES_IV)
        padded_data = pad(plaintext.encode("utf-8"), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted_data).decode("utf-8")
    except Exception as e:
        logger.error(f"AES encryption failed: {str(e)}")
        raise

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "edge-server"}), 200

# Global state to track client weights
client_weights_buffer = []
REQUIRED_CLIENTS = 2

@app.route('/uploadWeights', methods=['POST'])
def upload_weights():
    global client_weights_buffer
    
    try:
        data = request.get_json()
        
        if not data or "weights" not in data:
            return jsonify({"error": "Missing 'weights' field"}), 400

        # Direct access to plaintext weights (with differential privacy applied)
        weights_data = data["weights"]
        client_id = len(client_weights_buffer) + 1
        
        logger.info(f"Received weights from client {client_id}")
        
        if not isinstance(weights_data, list):
            return jsonify({"error": "Weights data must be a list"}), 400

        client_weights_buffer.append({
            "client_id": client_id,
            "weights": weights_data,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Stored weights from client {client_id}. Total: {len(client_weights_buffer)}/{REQUIRED_CLIENTS}")
        
        # Check if we have enough clients to trigger aggregation
        if len(client_weights_buffer) >= REQUIRED_CLIENTS:
            logger.info("Required clients reached. Forwarding to central server...")
            
            # Forward to central server
            success = forward_weights_to_central()
            
            if success:
                # Clear buffer after successful forwarding
                client_weights_buffer = []
                
                return jsonify({
                    "status": "success",
                    "message": "Weights received and forwarded to central server",
                    "client_id": client_id,
                    "aggregation_triggered": True
                }), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to forward weights to central server",
                    "client_id": client_id,
                    "aggregation_triggered": False
                }), 500
        else:
            return jsonify({
                "status": "success",
                "message": f"Weights received. Waiting for {REQUIRED_CLIENTS - len(client_weights_buffer)} more clients",
                "client_id": client_id,
                "aggregation_triggered": False
            }), 200
        
    except Exception as e:
        logger.error(f"Error in upload_weights: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
def forward_weights_to_central():
    """Forward collected weights to central server"""
    try:
        # Prepare payload with all client weights
        payload_data = {
            "clients": client_weights_buffer,
            "total_clients": len(client_weights_buffer)
        }
        
        # Single AES encryption for server communication
        aes_encrypted = encrypt_aes(json.dumps(payload_data))
        
        payload = {"encrypted_data": aes_encrypted}
        
        logger.info(f"Forwarding weights from {len(client_weights_buffer)} clients to central server")
        
        response = requests.post(
            f"{CENTRAL_SERVER_URL}/uploadWeights",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            logger.info("Successfully forwarded weights to central server")
            return True
        else:
            logger.error(f"Central server error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error forwarding weights: {str(e)}")
        return False



@app.route('/status', methods=['GET'])
def get_status():
    """Get edge server status and central server connectivity"""
    try:
        # Test connection to central server
        response = requests.get(f"{CENTRAL_SERVER_URL}/health", timeout=5)
        central_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        central_status = "disconnected"
    
    return jsonify({
        "edge_server": "running",
        "central_server": central_status,
        "central_url": CENTRAL_SERVER_URL
    }), 200

@app.route('/receiveGlobalModel', methods=['POST'])
def receive_global_model():
    """Receive global model from central server and store for clients"""
    global global_model_storage
    
    try:
        data = request.get_json()
        
        if not data or "encrypted_data" not in data:
            return jsonify({"error": "Missing 'encrypted_data' field"}), 400
        
        # Decrypt using server AES key (communication with central server)
        global_model_json = decrypt_aes(data["encrypted_data"])
        global_model_data = json.loads(global_model_json)
        
        # Store global model for plaintext delivery to clients
        global_model_storage = global_model_data
        
        logger.info(f"Received and stored global model from central server. Weights: {len(global_model_data.get('weights', []))}")
        
        return jsonify({
            "status": "success",
            "message": "Global model received and stored successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error receiving global model: {str(e)}")
        return jsonify({"error": str(e)}), 500
       
@app.route('/getGlobalModel', methods=['GET'])
def get_global_model():
    """Provide global model to clients"""
    try:
        if global_model_storage is None:
            logger.info("Global model requested but not available yet")
            return jsonify({"error": "Global model not available"}), 404
        
        logger.info("Providing global model to client")
        
        # Send plaintext JSON directly
        return jsonify({
            "status": "success",
            "model": global_model_storage  # Direct plaintext
        }), 200
        
    except Exception as e:
        logger.error(f"Error providing global model: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting Edge Server on port {port}")
    logger.info(f"Central Server URL: {CENTRAL_SERVER_URL}")
    app.run(host='0.0.0.0', port=port, debug=False)