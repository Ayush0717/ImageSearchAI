import os
import open_clip
import torch
from PIL import Image
import chromadb
import numpy as np

# Configurations
DATASET_PATH = "/Users/ayushgoel/MindFlixAi/BPLD Dataset"  # Change this to your dataset path
TEST_IMAGE_PATH = "/Users/ayushgoel/MindFlixAi/BPLD Dataset/Anthracnose 230/1a.jpg"  # Add a sample image for testing
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize CLIP Model
clip_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
clip_model.to(DEVICE)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection(name="image_search")

# Step 1: Load and Process Image
def load_and_preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Step 2: Generate Embeddings
def generate_clip_embedding(image_tensor):
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_tensor).float().cpu().numpy().flatten().tolist()
    return image_embedding

# Step 3: Store Embeddings in ChromaDB
def store_embedding(image_name, image_embedding):
    try:
        if image_embedding is None or len(image_embedding) == 0:
            print(f"❌ Embedding is None or empty for {image_name}")
            return
        
        collection.add(
            ids=[image_name],
            embeddings=[image_embedding],
            metadatas=[{"image_name": image_name}]
        )
        print(f"✅ Successfully stored embedding for {image_name}")
    except Exception as e:
        print(f"❌ Error storing embedding in ChromaDB: {e}")

# Step 4: Retrieve and Validate Embeddings
def retrieve_embedding(image_name):
    query = collection.get(ids=[image_name], include=["embeddings"])
    retrieved_embedding = query.get("embeddings", [None])[0]

    if retrieved_embedding is None:
        print(f"❌ No embedding found for {image_name}")
    else:
        print(f"✅ Retrieved embedding for {image_name}, Sample: {retrieved_embedding[:5]}")

# Step 5: Run Tests
def test_pipeline():
    print("\n🔹 Testing CLIP + ChromaDB Pipeline...\n")
    
    # Load Image
    image_tensor = load_and_preprocess_image(TEST_IMAGE_PATH)
    if image_tensor is None:
        return
    
    # Generate Embedding
    embedding = generate_clip_embedding(image_tensor)
    print(f"🔹 Generated Embedding Shape: {len(embedding)}\n")
    
    # Store in ChromaDB
    store_embedding("test_image", embedding)
    
    # Retrieve and Check
    retrieve_embedding("test_image")

# Run the test
test_pipeline()
