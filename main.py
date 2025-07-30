import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="model-fyp.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Model loaded successfully. Input shape: {input_details[0]['shape']}")
    logger.info(f"Input dtype: {input_details[0]['dtype']}")
    logger.info(f"Output shape: {output_details[0]['shape']}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','dot','space'
]

# Initialize MediaPipe with settings optimized for accuracy
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.7,  # Higher for stability
    min_tracking_confidence=0.5,   # Lower for responsiveness
    model_complexity=1             # Higher complexity for better accuracy
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of active connections
active_connections = set()
frame_count = {}  # Track frames per connection for debugging

async def process_frame(frame_data: str, connection_id: str) -> str:
    """Process a single frame and return prediction with proper preprocessing"""
    try:
        frame_count[connection_id] = frame_count.get(connection_id, 0) + 1
        
        # Decode base64 image
        if "," in frame_data:
            image_bytes = base64.b64decode(frame_data.split(",")[1])
        else:
            image_bytes = base64.b64decode(frame_data)
        
        # Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.warning("Failed to decode image")
            return "ERROR: Failed to decode image"
        
        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Check if any hands are detected
        if results.multi_hand_landmarks:            
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks using the SAME preprocessing as training
                x_coords = []
                y_coords = []
                
                # Get all landmark coordinates
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
                
                # Method 1: Normalized relative to bounding box (most common)
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # Avoid division by zero
                width = max_x - min_x if max_x - min_x > 0 else 1
                height = max_y - min_y if max_y - min_y > 0 else 1
                
                normalized_landmarks = []
                for lm in hand_landmarks.landmark:
                    # Normalize relative to bounding box
                    norm_x = (lm.x - min_x) / width
                    norm_y = (lm.y - min_y) / height
                    normalized_landmarks.extend([norm_x, norm_y])
                
                # Method 2: Alternative - distance-based normalization
                # Calculate distances from wrist (landmark 0)
                wrist_x, wrist_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                distance_landmarks = []
                for lm in hand_landmarks.landmark:
                    dist_x = lm.x - wrist_x
                    dist_y = lm.y - wrist_y
                    distance_landmarks.extend([dist_x, dist_y])
                
                # Method 3: Raw coordinates (as backup)
                raw_landmarks = []
                for lm in hand_landmarks.landmark:
                    raw_landmarks.extend([lm.x, lm.y])
                
                # Try all three methods - use the one that works best
                methods = [
                    ("normalized", normalized_landmarks),
                    ("distance", distance_landmarks), 
                    ("raw", raw_landmarks)
                ]
                
                best_prediction = None
                best_confidence = 0
                
                for method_name, landmarks in methods:
                    try:
                        # Prepare input for model
                        input_len = input_details[0]['shape'][1]
                        
                        # Handle input size mismatch
                        if len(landmarks) < input_len:
                            landmarks_padded = landmarks + [0.0] * (input_len - len(landmarks))
                        elif len(landmarks) > input_len:
                            landmarks_padded = landmarks[:input_len]
                        else:
                            landmarks_padded = landmarks
                        
                        # Convert to numpy array
                        input_data = np.array([landmarks_padded], dtype=np.float32)
                        
                        # Run inference
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        
                        # Get prediction
                        output = interpreter.get_tensor(output_details[0]['index'])[0]
                        predicted_index = int(np.argmax(output))
                        confidence = float(np.max(output))
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_prediction = (labels[predicted_index], confidence, method_name)
                            
                    except Exception as method_error:
                        logger.warning(f"Method {method_name} failed: {method_error}")
                        continue
                
                if best_prediction:
                    predicted_char, confidence, method = best_prediction
                    logger.info(f"Best prediction: {predicted_char} (confidence: {confidence:.3f}, method: {method})")
                    
                    # Use higher confidence threshold for better accuracy
                    if confidence > 0.6:  # Increased threshold
                        return f"{predicted_char}"
                    elif confidence > 0.4:
                        return f"{predicted_char}?"  # Show uncertainty
                    else:
                        return "Low confidence"
                else:
                    return "Processing failed"
        else:
            return ""  # No hand detected - return empty for cleaner UI
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"ERROR: {str(e)}"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    active_connections.add(websocket)
    frame_count[connection_id] = 0
    logger.info(f"WebSocket connected. Active connections: {len(active_connections)}")
    
    try:
        while True:
            try:
                # Receive data
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                logger.info(f"Received data of length: {len(data)}")
                
                # Process the frame
                result = await process_frame(data, connection_id)
                
                # Send result back
                await websocket.send_text(result)
                logger.info(f"Sent result: {result}")
                
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                await websocket.send_text("TIMEOUT")
                    
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                try:
                    await websocket.send_text(f"ERROR: {str(e)}")
                except:
                    break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        active_connections.discard(websocket)
        if connection_id in frame_count:
            del frame_count[connection_id]
        logger.info(f"WebSocket connection closed. Active connections: {len(active_connections)}")

@app.websocket("/testws")
async def test_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("Test WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        logger.info("Test WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")

@app.get("/")
async def root():
    return {
        "message": "Sign Detection API", 
        "active_connections": len(active_connections),
        "model_loaded": True,
        "supported_signs": len(labels),
        "model_input_shape": input_details[0]['shape'],
        "model_output_shape": output_details[0]['shape']
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(active_connections),
        "labels": labels,
        "frame_counts": frame_count
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Sign Detection API server...")
    uvicorn.run(app, host="0.0.0.0", port=8006)