from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad
import logging
from datetime import datetime
import requests
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration from environment variables
AES_KEY = bytes.fromhex(os.getenv('AES_KEY', '000102030405060708090a0b0c0d0e0f'))
AES_IV = bytes.fromhex(os.getenv('AES_IV', '101112131415161718191a1b1c1d1e1f'))

# Global state
client_updates = []
global_model_weights = None
participants_count = 0

def decrypt_aes(encrypted_text):
    """Decrypt data using AES-256-CBC"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_text)
        cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        plaintext = unpad(decrypted_bytes, AES.block_size).decode("utf-8")
        return plaintext
    except Exception as e:
        logger.error(f"AES decryption failed: {str(e)}")
        raise
def encrypt_aes(plaintext):
    """Encrypt data using AES-256-CBC"""
    try:
        cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
        padded_data = pad(plaintext.encode("utf-8"), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted_data).decode("utf-8")
    except Exception as e:
        logger.error(f"AES encryption failed: {str(e)}")
        raise

def validate_weights_structure(weights_list):
    """Validate the structure of weights data"""
    if not isinstance(weights_list, list):
        return False
    
    for weight in weights_list:
        if not isinstance(weight, dict):
            return False
        required_keys = ['layerIndex', 'weightIndex', 'shape', 'values']
        if not all(key in weight for key in required_keys):
            return False
        if not isinstance(weight['values'], list):
            return False
        if not isinstance(weight['shape'], list):
            return False
    
    return True

def federated_averaging(client_weights_list):
    """Perform federated averaging with proper weight structure preservation"""
    if not client_weights_list:
        raise ValueError("No client weights to aggregate")
    
    num_clients = len(client_weights_list)
    logger.info(f"Aggregating weights from {num_clients} clients")
    
    # Initialize aggregated weights
    aggregated_weights = []
    
    # Sort weights by layerIndex to ensure consistent ordering
    sorted_weights = sorted(client_weights_list[0], key=lambda x: x["layerIndex"])
    
    for weight_info in sorted_weights:
        # Initialize with zeros
        aggregated_values = np.zeros(len(weight_info["values"]), dtype=np.float32)
        
        # Average across all clients
        for client_weights in client_weights_list:
            # Find matching weight by layerIndex
            matching_weight = next(w for w in client_weights if w["layerIndex"] == weight_info["layerIndex"])
            client_values = np.array(matching_weight["values"], dtype=np.float32)
            aggregated_values += client_values
        
        # Average by number of clients
        aggregated_values = aggregated_values / num_clients
        
        aggregated_weights.append({
            "layerIndex": weight_info["layerIndex"],
            "weightIndex": weight_info["weightIndex"],
            "shape": weight_info["shape"],
            "values": aggregated_values.tolist()
        })
    
    logger.info(f"Aggregation completed for {len(aggregated_weights)} weight tensors")
    return aggregated_weights

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "central-server",
        "participants": len(client_updates),
        "model_initialized": global_model_weights is not None
    }), 200

@app.route('/uploadWeights', methods=['POST'])
def upload_weights():
    """Receive encrypted weights from edge server and auto-aggregate"""
    global client_updates, participants_count, global_model_weights
    
    try:
        data = request.get_json()
        
        if not data or "encrypted_data" not in data:
            return jsonify({"error": "Missing 'encrypted_data' field"}), 400

        # Single AES decryption
        payload_json = decrypt_aes(data["encrypted_data"])
        payload_data = json.loads(payload_json)
        
        # Extract client weights
        clients_data = payload_data["clients"]
        total_clients = payload_data["total_clients"]
        
        logger.info(f"Received weights from {total_clients} clients via edge server")
        
        # Validate and store all client weights
        client_weights_list = []
        for client_data in clients_data:
            weights_list = client_data["weights"]
            
            if not validate_weights_structure(weights_list):
                return jsonify({"error": f"Invalid weights structure for client {client_data['client_id']}"}), 400
            
            client_weights_list.append(weights_list)
            participants_count += 1
        
        # Automatically perform aggregation
        logger.info(f"Starting automatic aggregation with {len(client_weights_list)} participants")
        
        # Perform federated averaging
        global_model_weights = federated_averaging(client_weights_list)
        
        # Prepare aggregation info
        aggregation_info = {
            "timestamp": datetime.now().isoformat(),
            "participants": len(client_weights_list),
            "total_parameters": sum(len(w["values"]) for w in global_model_weights)
        }
        
        logger.info(f"Aggregation completed. Participants: {len(client_weights_list)}")
        
        # Send global model back to edge server
        success = send_global_model_to_edge(global_model_weights, aggregation_info)
        
        if success:
            return jsonify({
                "message": "Weights received and aggregated successfully",
                "participants": len(client_weights_list),
                "aggregation_info": aggregation_info,
                "global_model_sent": True
            }), 200
        else:
            return jsonify({
                "message": "Aggregation completed but failed to send global model",
                "participants": len(client_weights_list),
                "aggregation_info": aggregation_info,
                "global_model_sent": False
            }), 207

    except Exception as e:
        logger.error(f"Error in upload_weights: {str(e)}")
        return jsonify({"error": f"Failed to process weights: {str(e)}"}), 500

def send_global_model_to_edge(weights, aggregation_info):
    """Send global model back to edge server"""
    try:
        # Prepare global model data
        global_model_data = {
            "weights": weights,
            "aggregation_info": aggregation_info
        }
        
        # Single AES encryption
        aes_encrypted = encrypt_aes(json.dumps(global_model_data))
        
        payload = {"encrypted_data": aes_encrypted}
        
        # Send to edge server
        edge_server_url = os.getenv('EDGE_SERVER_URL', 'http://localhost:8000')
        response = requests.post(
            f"{edge_server_url}/receiveGlobalModel",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info("Successfully sent global model to edge server")
            return True
        else:
            logger.error(f"Failed to send global model to edge server: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending global model to edge server: {str(e)}")
        return False

def encrypt_aes(plaintext):
    """Encrypt data using AES-256-CBC"""
    try:
        from Crypto.Util.Padding import pad
        cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
        padded_data = pad(plaintext.encode("utf-8"), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted_data).decode("utf-8")
    except Exception as e:
        logger.error(f"AES encryption failed: {str(e)}")
        raise

@app.route('/globalWeights', methods=['GET'])
def get_global_weights():
    """Get the current global model weights"""
    try:
        if global_model_weights is None:
            return jsonify({"error": "Global model not yet initialized"}), 404
        
        return jsonify({
            "global_weights": global_model_weights,
            "participants": participants_count,
            "total_parameters": sum(len(w["values"]) for w in global_model_weights)
        }), 200

    except Exception as e:
        logger.error(f"Error in get_global_weights: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get central server status"""
    return jsonify({
        "status": "running",
        "pending_updates": len(client_updates),
        "total_participants": participants_count,
        "global_model_ready": global_model_weights is not None,
        "last_aggregation": datetime.now().isoformat() if global_model_weights else None
    }), 200

@app.route('/reset', methods=['POST'])
def reset_server():
    """Reset server state (for testing purposes)"""
    global client_updates, global_model_weights, participants_count
    
    client_updates = []
    global_model_weights = None
    participants_count = 0
    
    logger.info("Server state reset")
    return jsonify({"message": "Server state reset successfully"}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8001))
    logger.info(f"Starting Central Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)