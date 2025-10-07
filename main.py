"""
Skin Disease Classifier API
FastAPI server for skin lesion classification
Integrated with SkinAnalyzer mobile app
"""

import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import torch.nn as nn
import timm
import gdown
import os
from torchvision import transforms
from PIL import Image
ML_AVAILABLE = True

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
MODEL_PATH = "best_model_cnn.pth"  # Update this path to your model
MODEL_URL = "https://drive.google.com/uc?id=1LS0SCSye7PUrL1bL6qt4MOoxivxbRzX2"
NUM_CLASSES = 7

if not os.path.exists(MODEL_PATH):
    print("Downloading from Google Drive")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Download completed")
else:
    print("Model file already exists. Skipping download.")    

# Disease information
DISEASE_INFO = {
    0: {
        "code": "akiec",
        "name": "Actinic Keratoses",
        "description": "Pre-cancerous skin growths caused by sun damage",
        "severity": "Medium",
        "recommendations": [
            "Consult a dermatologist for proper diagnosis",
            "Use broad-spectrum sunscreen daily",
            "Avoid excessive sun exposure",
            "Consider cryotherapy or topical treatments"
        ]
    },
    1: {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "description": "Most common form of skin cancer, rarely spreads",
        "severity": "High",
        "recommendations": [
            "Immediate dermatologist consultation required",
            "Surgical removal is often necessary",
            "Regular skin cancer screenings",
            "Sun protection is crucial"
        ]
    },
    2: {
        "code": "bkl",
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growths, often age-related",
        "severity": "Low",
        "recommendations": [
            "Usually no treatment needed",
            "Monitor for changes in appearance",
            "Can be removed for cosmetic reasons",
            "Protect from sun exposure"
        ]
    },
    3: {
        "code": "df",
        "name": "Dermatofibroma",
        "description": "Harmless hard bump, often after insect bite",
        "severity": "Low",
        "recommendations": [
            "No treatment necessary",
            "Can be surgically removed if bothersome",
            "Usually stable and harmless",
            "Monitor for any changes"
        ]
    },
    4: {
        "code": "mel",
        "name": "Melanoma",
        "description": "Serious form of skin cancer that can spread",
        "severity": "Critical",
        "recommendations": [
            "URGENT dermatologist consultation",
            "Early detection is crucial for treatment",
            "Surgical excision typically required",
            "Regular full-body skin exams"
        ]
    },
    5: {
        "code": "nv",
        "name": "Melanocytic Nevi",
        "description": "Common moles, usually harmless",
        "severity": "Low",
        "recommendations": [
            "Monitor for ABCDE changes (Asymmetry, Border, Color, Diameter, Evolution)",
            "Regular self-examinations",
            "Sun protection to prevent changes",
            "Remove if changing or suspicious"
        ]
    },
    6: {
        "code": "vasc",
        "name": "Vascular Lesions",
        "description": "Blood vessel-related skin conditions",
        "severity": "Low",
        "recommendations": [
            "Consult dermatologist for diagnosis",
            "Laser treatment options available",
            "Usually harmless but can be treated",
            "Monitor for changes in size or color"
        ]
    }
}

# =============================================================================
# MODEL LOADING (Optional)
# =============================================================================

if ML_AVAILABLE:
    try:
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {device}")

        # Image transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Try to load model
        try:
            print("🤖 Loading skin disease model...")
            model = timm.create_model("efficientnet_b3", pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

            # Load checkpoint (adjust this based on your model format)
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            model.to(device)
            print("✅ Model loaded successfully!")

        except FileNotFoundError:
            print(f"⚠️  Model file not found at {MODEL_PATH}. Using mock mode.")
            model = None
        except Exception as e:
            print(f"⚠️  Error loading model: {e}. Using mock mode.")
            model = None

    except Exception as e:
        print(f"⚠️  ML setup failed: {e}. Using mock mode.")
        model = None
else:
    print("🔧 Running in mock mode (no ML dependencies)")

# =============================================================================
# DATA MODELS
# =============================================================================

class User(BaseModel):
    id: str
    email: str
    name: str
    skin_type: str

class TreatmentItem(BaseModel):
    id: str
    name: str
    dosage: str
    frequency: str
    duration: str
    notes: str
    completed: bool

class TreatmentUpdate(BaseModel):
    completed: bool

# =============================================================================
# MOCK DATA
# =============================================================================

mock_user = User(
    id="user1",
    email="john.doe@email.com",
    name="John Doe",
    skin_type="Sensitive"
)

mock_treatments = [
    TreatmentItem(
        id="1",
        name="Hydrocortisone Cream 1%",
        dosage="Apply thin layer",
        frequency="Twice daily",
        duration="2 weeks",
        notes="Apply to affected areas after cleansing",
        completed=False
    ),
    TreatmentItem(
        id="2",
        name="Cetirizine (Antihistamine)",
        dosage="10mg",
        frequency="Once daily",
        duration="1 week",
        notes="Take with or without food",
        completed=False
    ),
    TreatmentItem(
        id="3",
        name="Moisturizing Lotion",
        dosage="Apply generously",
        frequency="Daily",
        duration="Ongoing",
        notes="Use fragrance-free moisturizer",
        completed=False
    )

]

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Skin Disease Classifier API",
    description="API for skin lesion classification and treatment management",
    version="2.0.0"
)

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def preprocess_image(image):
    """Preprocess image for model inference"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0).to(device)

def analyze_with_model(image):
    """Analyze image using the ML model"""
    with torch.no_grad():
        img_tensor = preprocess_image(image)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    return pred_idx, confidence, probs

def analyze_with_mock(image):
    """Mock analysis for testing"""
    import random
    # Simulate processing time
    import time
    time.sleep(1)

    # Pick random disease
    pred_idx = random.randint(0, NUM_CLASSES - 1)
    confidence = round(random.uniform(0.75, 0.95), 4)

    # Mock probabilities
    probs = torch.rand(NUM_CLASSES) if torch else [0.1] * NUM_CLASSES

    return pred_idx, confidence, probs

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Skin Disease Classifier API",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "mode": "ML" if model else "Mock",
        "endpoints": {
            "/api/v1/analysis/upload": "POST - Upload skin image for analysis",
            "/api/v1/health": "GET - API health status",
            "/api/v1/users/me": "GET - Current user info",
            "/api/v1/treatments": "GET - Treatment plans"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "mode": "ML" if model else "Mock"
    }

@app.get("/api/v1/users/me")
async def get_current_user():
    """Get current user profile"""
    return mock_user

@app.get("/api/v1/treatments")
async def get_treatments():
    """Get user's treatment plans"""
    return mock_treatments

@app.put("/api/v1/treatments/{treatment_id}")
async def update_treatment(treatment_id: str, update_data: TreatmentUpdate):
    """Update treatment completion status"""
    for treatment in mock_treatments:
        if treatment.id == treatment_id:
            treatment.completed = update_data.completed
            return treatment
    raise HTTPException(status_code=404, detail="Treatment not found")

@app.get("/api/v1/analysis/history")
async def get_analysis_history(status: Optional[str] = None, search: Optional[str] = None):
    """Get analysis history (mock data for now)"""
    # Return empty array - in production, this would query a database
    return []

@app.get("/api/v1/diseases")
async def get_diseases():
    """Get information about all detectable diseases"""
    return {
        "diseases": DISEASE_INFO,
        "count": NUM_CLASSES
    }

@app.post("/api/v1/analysis/upload")
async def analyze_skin(image: UploadFile = File(...), metadata: str = Form("{}")):
    """
    Analyze a skin image and return disease classification
    Works with both real model and mock data
    """
    try:
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file type
        allowed_types = {".jpg", ".jpeg", ".png", ".bmp"}
        file_ext = Path(image.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Use: {', '.join(allowed_types)}"
            )

        # Read image
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if not ML_AVAILABLE or Image is None:
            raise HTTPException(
                status_code=503,
                detail="Image processing not available. Check server dependencies."
            )

        # Load image
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Perform analysis
        if model is not None:
            print("🔬 Using ML model for analysis...")
            pred_idx, confidence, probs = analyze_with_model(pil_image)
        else:
            print("🎭 Using mock analysis...")
            pred_idx, confidence, probs = analyze_with_mock(pil_image)

        # Get disease information
        disease_info = DISEASE_INFO[pred_idx]

        # Prepare response
        analysis_id = str(uuid.uuid4())

        response = {
            "id": analysis_id,
            "condition": disease_info["code"],
            "condition_name": disease_info["name"],
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "severity": disease_info["severity"],
            "description": disease_info["description"],
            "recommendations": disease_info["recommendations"],
            "timestamp": datetime.now().isoformat(),
            "model_used": "ML" if model else "Mock"
        }

        print(f"✅ Analysis complete: {disease_info['name']} ({confidence:.1%})")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# =============================================================================
# APPLICATION EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 50)
    print("🚀 Skin Disease Classifier API Started")
    print(f"🔧 Mode: {'ML' if model else 'Mock'}")
    print(f"📊 Diseases: {NUM_CLASSES}")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("👋 Shutting down Skin Disease Classifier API")

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True)