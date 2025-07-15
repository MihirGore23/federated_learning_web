const CONFIG = {
    EDGE_SERVER_URL: 'http://localhost:8000', 
    IMAGE_SIZE: 96,
    BATCH_SIZE: 8,
    EPOCHS: 7,
    LEARNING_RATE: 5e-5,
};

let appState = {
    model: null,
    dataset: [],
    isTraining: false,
    trainingComplete: false,
    currentEpoch: 0,
    logs: []
};

const elements = {
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    fileCount: document.getElementById('file-count'),
    trainBtn: document.getElementById('train-btn'),
    progressFill: document.getElementById('progress-fill'),
    progressPercent: document.getElementById('progress-percent'),
    epochText: document.getElementById('epoch-text'),
    accuracy: document.getElementById('accuracy'),
    loss: document.getElementById('loss'),
    finalValidationAccuracy: document.getElementById('final-validation-accuracy'),
    logContainer: document.getElementById('log-container')
};

document.addEventListener('DOMContentLoaded', async () => {
    await initializeTensorFlow();
    setupEventListeners();
    logMessage('System initialized successfully', 'success');
});

function addMinimalNoise(weights, epsilon = 1e6) {
    // With epsilon = 1e6, noise should be practically zero
    const noiseScale = 1.0 / epsilon; // This gives 1e-6 noise scale
    
    logMessage(`Applying minimal noise (σ=${noiseScale.toExponential(2)})...`, 'info');
    
    const noisyWeights = weights.map(weightInfo => {
        const values = weightInfo.values;
        const noisyValues = values.map(val => {
            // Box-Muller transform for Gaussian noise
            const u1 = Math.random();
            const u2 = Math.random();
            const gaussianNoise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            return val + gaussianNoise * noiseScale;
        });

        return {
            ...weightInfo,
            values: noisyValues
        };
    });

    return noisyWeights;
}


async function initializeTensorFlow() {
    try {
        await tf.ready();
        logMessage(`TensorFlow.js loaded. Backend: ${tf.getBackend()}`, 'info');
        updateStatus('Ready', 'ready');
    } catch (error) {
        logMessage(`Failed to initialize TensorFlow.js: ${error.message}`, 'error');
        updateStatus('Error', 'error');
    }
}

function setupEventListeners() {
    // File upload
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Only train button (aggregate button removed)
    elements.trainBtn.addEventListener('click', startTraining);
}

function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    processFiles(files);
}

function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    const files = Array.from(event.dataTransfer.files);
    processFiles(files);
}

async function processFiles(files) {
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        logMessage('No valid image files selected', 'warning');
        return;
    }

    logMessage(`Processing ${imageFiles.length} images...`, 'info');
    
    try {
        appState.dataset = [];
        let labelCounts = { 0: 0, 1: 0 };
        let invalidFiles = [];
        
        for (const file of imageFiles) {
            const imageData = await loadImageFile(file);
            if (imageData) {
                // Extract label from filename pattern: {imagenumber}_{label}
                const labelMatch = file.name.match(/^\d+_(\d+)(?:\.\w+)?$/);
                
                if (labelMatch) {
                    const label = parseInt(labelMatch[1], 10);
                    appState.dataset.push({ 
                        image: imageData, 
                        label: label,
                        filename: file.name
                    });
                    
                    // Count labels for logging
                    if (label === 0 || label === 1) {
                        labelCounts[label]++;
                    }
                    
                    logMessage(`✓ Loaded ${file.name} with label ${label}`, 'info');
                } else {
                    invalidFiles.push(file.name);
                    logMessage(`⚠ Could not extract label from filename: ${file.name}`, 'warning');
                }
            }
        }

        // Show summary
        elements.fileCount.textContent = `${appState.dataset.length} images loaded`;
        elements.fileCount.classList.add('show');
        elements.trainBtn.disabled = appState.dataset.length === 0;
        
        logMessage(`Successfully loaded ${appState.dataset.length} images`, 'success');
        logMessage(`Label distribution: Class 0: ${labelCounts[0]}, Class 1: ${labelCounts[1]}`, 'info');
        
        // Show warnings for invalid files
        if (invalidFiles.length > 0) {
            logMessage(`${invalidFiles.length} files had invalid naming format`, 'warning');
            logMessage(`Expected format: {imagenumber}_{label}.{extension} (e.g., 0_1.jpg, 3_0.png)`, 'warning');
        }
        
        // Check for balanced dataset
        const totalLabeled = labelCounts[0] + labelCounts[1];
        if (totalLabeled !== appState.dataset.length) {
            logMessage(`Warning: ${appState.dataset.length - totalLabeled} images have labels other than 0 or 1`, 'warning');
        }
        
        // Dataset balance check
        if (labelCounts[0] > 0 && labelCounts[1] > 0) {
            const ratio = Math.min(labelCounts[0], labelCounts[1]) / Math.max(labelCounts[0], labelCounts[1]);
            if (ratio < 0.3) {
                logMessage(`Dataset is imbalanced. Consider adding more samples of the minority class.`, 'warning');
            }
        }
        
    } catch (error) {
        logMessage(`Error processing files: ${error.message}`, 'error');
    }
}

async function loadImageFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                try {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = CONFIG.IMAGE_SIZE;
                    canvas.height = CONFIG.IMAGE_SIZE;
                    
                    ctx.drawImage(img, 0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
                    const imageData = ctx.getImageData(0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
                    const tensor = tf.browser.fromPixels(imageData).div(255.0);
                    resolve(tensor);
                } catch (error) {
                    reject(error);
                }
            };
            img.onerror = reject;
            img.src = event.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function createModel() {
    try {
        logMessage('Loading MobileNetV2 model for federated learning...', 'info');

        // Load the properly converted model
        const model = await tf.loadLayersModel('./model/model.json');

        // Unfreeze last 42 layers for training
        const totalLayers = model.layers.length;
        for (let i = 0; i < totalLayers; i++) {
            model.layers[i].trainable = (i >= totalLayers - 42);
        }

        // Exponential decay learning rate schedule
        const initialLR = 7e-5;
        const decaySteps = 1000;
        const decayRate = 0.9;
        const lrSchedule = exponentialDecay(initialLR, decaySteps, decayRate);
        let globalStep = 0;
        // Custom Adam optimizer with dynamic learning rate
        const optimizer = tf.train.adam(lrSchedule(globalStep));

        // Compile with L2 regularization for new layers (if any are added)
        // Note: L2 regularization cannot be retroactively added to loaded layers
        model.compile({
            optimizer: optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        logMessage('Model loaded and compiled successfully', 'success');
        logMessage(`Input shape: [${model.inputs[0].shape.join(', ')}]`, 'info');
        logMessage(`Output shape: [${model.outputs[0].shape.join(', ')}]`, 'info');

        // Optionally, if you add new layers, add L2 regularization like this:
        // new tf.layers.dense({units: ..., activation: ..., kernelRegularizer: tf.regularizers.l2({l2: 0.005})})

        return model;
    } catch (error) {
        logMessage(`Error loading model: ${error.message}`, 'error');
        console.error('Detailed error:', error);
        // Fallback: create a simple model if loading fails
        logMessage('Creating fallback model...', 'warning');
        return createFallbackModel();
    }
}

async function startTraining() {
    if (appState.dataset.length === 0) {
        logMessage('No dataset loaded', 'warning');
        return;
    }

    try {
        appState.isTraining = true;
        elements.trainBtn.disabled = true;
        elements.trainBtn.innerHTML = '<div class="loading"></div> Training...';
        updateStatus('Training in progress', 'training');

        // Create model
        appState.model = await createModel();
        
        // Prepare dataset
        const { xs, ys } = prepareDataset();
        
        // Train model
        await trainModel(xs, ys);
        
        // Extract and send weights
        await sendWeights();
        
        appState.trainingComplete = true;
        logMessage('Waiting for global model from server...', 'info');
        elements.trainBtn.innerHTML = 'Training Complete';
        updateStatus('Training completed', 'ready');
        
        logMessage('Local training completed successfully', 'success');
        
        // Clean up tensors
        xs.dispose();
        ys.dispose();
        
    } catch (error) {
        logMessage(`Training failed: ${error.message}`, 'error');
        updateStatus('Training failed', 'error');
        elements.trainBtn.innerHTML = 'Start Local Training';
        elements.trainBtn.disabled = false;
        appState.isTraining = false;
    }
}

function prepareDataset() {
    const images = appState.dataset.map(item => item.image);
    const labels = appState.dataset.map(item => item.label); // Use actual labels (0 or 1)
    
    const xs = tf.stack(images);
    const ys = tf.tensor2d(labels.map(label => [label])); // Shape: [batchSize, 1]
    
    // Log dataset statistics
    const labelCounts = labels.reduce((acc, label) => {
        acc[label] = (acc[label] || 0) + 1;
        return acc;
    }, {});
    
    console.log('Dataset prepared:', {
        images: xs.shape,
        labels: ys.shape,
        sampleLabels: labels.slice(0, 5),
        labelDistribution: labelCounts
    });
    
    logMessage(`Dataset prepared: ${labels.length} samples, Distribution: ${JSON.stringify(labelCounts)}`, 'info');
    
    return { xs, ys };
}

async function trainModel(xs, ys) {
    logMessage('Starting model training...', 'info');
    
    const validationSplit = 0.2;
    
    await appState.model.fit(xs, ys, {
        epochs: CONFIG.EPOCHS,
        batchSize: CONFIG.BATCH_SIZE,
        validationSplit,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                appState.currentEpoch = epoch + 1;
                const progress = ((epoch + 1) / CONFIG.EPOCHS) * 100;
                
                elements.progressFill.style.width = `${progress}%`;
                elements.progressPercent.textContent = `${Math.round(progress)}%`;
                elements.epochText.textContent = `Epoch ${epoch + 1}/${CONFIG.EPOCHS}`;
                elements.accuracy.textContent = `${(logs.acc * 100).toFixed(2)}%`;
                elements.loss.textContent = logs.loss.toFixed(4);
                
                logMessage(`Epoch ${epoch + 1}: acc=${(logs.acc * 100).toFixed(2)}%, loss=${logs.loss.toFixed(4)}, val_acc=${(logs.val_acc * 100).toFixed(2)}%`, 'info');
            }
        }
    });
}

async function sendWeights() {
    try {
        logMessage('Extracting model weights...', 'info');
        
        // Get weights in correct order and structure
        const modelWeights = appState.model.getWeights();
        const weights = [];
        
        // Group weights by layer properly
        for (let i = 0; i < modelWeights.length; i++) {
            const weight = modelWeights[i];
            weights.push({
                layerIndex: i,
                weightIndex: 0, // Keep simple indexing
                shape: Array.from(weight.shape),
                values: Array.from(weight.dataSync())
            });
        }

        logMessage(`Extracted ${weights.length} weight tensors`, 'info');
        
        // Apply minimal noise (or skip entirely for testing)
        const noisyWeights = addMinimalNoise(weights, 1e6);
        
        logMessage('Sending weights to edge server...', 'info');
        
        const response = await fetch(`${CONFIG.EDGE_SERVER_URL}/uploadWeights`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                weights: noisyWeights
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const result = await response.json();
        
        if (result.status === 'success') {
            if (result.aggregation_triggered) {
                logMessage('Weights sent successfully! Aggregation triggered.', 'success');
            } else {
                logMessage(`Weights sent successfully! ${result.message}`, 'info');
            }
            logMessage(`Client ID: ${result.client_id}`, 'info');
            
            // Start polling for global model
            pollForGlobalModel();
        } else {
            throw new Error(result.message || 'Unknown error');
        }
        
    } catch (error) {
        logMessage(`Failed to send weights: ${error.message}`, 'error');
        throw error;
    }
}

async function pollForGlobalModel() {
    const maxAttempts = 30; // Poll for 5 minutes (10 seconds * 30)
    let attempts = 0;
    
    logMessage('Waiting for global model...', 'info');
    
    const pollInterval = setInterval(async () => {
        attempts++;
        
        try {
            const response = await fetch(`${CONFIG.EDGE_SERVER_URL}/getGlobalModel`);
            
            if (response.ok) {
                const result = await response.json();
                clearInterval(pollInterval);
                
                logMessage('Global model received!', 'success');
                await applyGlobalModel(result.model);
                
            } else if (response.status === 404) {
                logMessage(`Waiting for global model... (${attempts}/${maxAttempts})`, 'info');
                
                if (attempts >= maxAttempts) {
                    clearInterval(pollInterval);
                    logMessage('Timeout waiting for global model', 'error');
                }
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
            
        } catch (error) {
            logMessage(`Polling attempt ${attempts}/${maxAttempts} failed`, 'info');
            
            if (attempts >= maxAttempts) {
                clearInterval(pollInterval);
                logMessage(`Failed to receive global model: ${error.message}`, 'error');
            }
        }
    }, 10000); // Poll every 10 seconds
}

async function applyGlobalModel(modelData) {
    try {
        logMessage('Processing global model...', 'info');
        
        const globalModelData = modelData.model || modelData;
        const globalWeights = globalModelData.weights;
        
        logMessage(`Received ${globalWeights.length} weight tensors`, 'info');
        
        // Create tensors in the same order as extraction
        const newWeights = [];
        
        // Sort by layerIndex to ensure correct order
        const sortedWeights = globalWeights.sort((a, b) => a.layerIndex - b.layerIndex);
        
        for (const weightInfo of sortedWeights) {
            const tensor = tf.tensor(weightInfo.values, weightInfo.shape);
            newWeights.push(tensor);
        }
        
        logMessage(`Created ${newWeights.length} tensors for model`, 'info');
        
        // Apply weights to current model
        appState.model.setWeights(newWeights);
        
        logMessage('Global model applied successfully!', 'success');
        
        // Validate immediately
        await validateOnLocalData();
        
        // Clean up tensors
        newWeights.forEach(weight => weight.dispose());
        
    } catch (error) {
        logMessage(`Failed to apply global model: ${error.message}`, 'error');
        console.error('Global model application error:', error);
        console.error('Model data structure:', modelData);
        elements.finalValidationAccuracy.textContent = 'Model application failed';
        elements.finalValidationAccuracy.style.color = '#ff6b6b';
    }
}

async function validateOnLocalData() {
    try {
        if (appState.dataset.length === 0) {
            logMessage('No local data available for validation', 'warning');
            elements.finalValidationAccuracy.textContent = 'No validation data';
            elements.finalValidationAccuracy.style.color = '#ff6b6b';
            return;
        }
        
        logMessage('Validating global model on local data...', 'info');
        
        const testData = appState.dataset;
        const testImages = testData.map(item => item.image);
        const testLabels = testData.map(item => item.label);
        
        const xs = tf.stack(testImages);
        const ys = tf.tensor2d(testLabels.map(label => [label]));
        
        // Get predictions
        const predictions = appState.model.predict(xs);
        const predictionData = await predictions.data();
        
        // Calculate accuracy with threshold 0.5
        let correct = 0;
        const threshold = 0.5;
        
        for (let i = 0; i < testLabels.length; i++) {
            const predicted = predictionData[i] > threshold ? 1 : 0;
            const actual = testLabels[i];
            if (predicted === actual) correct++;
        }
        
        const accuracy = correct / testLabels.length;
        const accuracyPercent = (accuracy * 100).toFixed(2);
        
        // Update UI
        elements.finalValidationAccuracy.textContent = `${accuracyPercent}%`;
        elements.finalValidationAccuracy.style.color = accuracy > 0.7 ? '#2ecc71' : '#f39c12';
        
        logMessage(`Global model validation accuracy: ${accuracyPercent}%`, 'success');
        
        // Log some predictions for debugging
        logMessage(`Sample predictions: ${Array.from(predictionData.slice(0, 5)).map(p => p.toFixed(3)).join(', ')}`, 'info');
        logMessage(`Sample labels: ${testLabels.slice(0, 5).join(', ')}`, 'info');
        
        // Clean up
        xs.dispose();
        ys.dispose();
        predictions.dispose();
        
    } catch (error) {
        logMessage(`Validation failed: ${error.message}`, 'error');
        console.error('Validation error:', error);
        elements.finalValidationAccuracy.textContent = 'Validation failed';
        elements.finalValidationAccuracy.style.color = '#ff6b6b';
    }
}



// Utility functions
function updateStatus(text, type) {
    elements.statusText.textContent = text;
    elements.statusDot.className = `status-dot ${type}`;
}

function logMessage(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-${type}">${message}</span>
    `;
    
    elements.logContainer.appendChild(logEntry);
    elements.logContainer.scrollTop = elements.logContainer.scrollHeight;
    
    appState.logs.push({ timestamp, message, type });
    
    // Keep only last 100 log entries
    if (appState.logs.length > 100) {
        appState.logs.shift();
        elements.logContainer.removeChild(elements.logContainer.firstChild);
    }
}

// Exponential decay learning rate function
function exponentialDecay(initialLR, decaySteps, decayRate) {
    return (step) => initialLR * Math.pow(decayRate, Math.floor(step / decaySteps));
}