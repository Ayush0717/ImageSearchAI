import os
import open_clip
import torch
from PIL import Image
import chromadb
import numpy as np
from tqdm import tqdm

# Configurations
DATASET_PATH = "/Users/ayushgoel/MindFlixAi/BPLD Dataset"  # Change this to your dataset path
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

# Load and Process Image
def load_and_preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        return image_tensor
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

# Generate Embeddings
def generate_clip_embedding(image_tensor):
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_tensor).float().cpu().numpy().flatten().tolist()
    return image_embedding

# Store Embeddings in ChromaDB
def store_embedding(image_name, image_embedding, category, image_path):
    try:
        if image_embedding is None or len(image_embedding) == 0:
            print(f"❌ Embedding is None or empty for {image_name}")
            return
        
        collection.add(
            ids=[image_name],
            embeddings=[image_embedding],
            metadatas=[{"category": category, "image_name": image_name, "image_path": image_path}]
        )
        print(f"✅ Stored: {image_name} (Category: {category})")
    except Exception as e:
        print(f"❌ Error storing {image_name} in ChromaDB: {e}")

# Process Dataset and Store Embeddings
def process_and_store_images(dataset_path):
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue  # Skip non-folder files

        print(f"\n📂 Processing Category: {category}")

        for image_name in tqdm(os.listdir(category_path)):
            image_path = os.path.join(category_path, image_name)

            # Load image
            image_tensor = load_and_preprocess_image(image_path)
            if image_tensor is None:
                continue

            # Generate embedding
            embedding = generate_clip_embedding(image_tensor)

            # Store in ChromaDB
            store_embedding(image_name, embedding, category, image_path)

    print("\n✅ Dataset processing complete. All embeddings stored in ChromaDB.")

# Search Similar Images in ChromaDB
def search_similar_images(query_image_path, top_k=5):
    print(f"\n🔎 Searching for images similar to: {query_image_path}")

    # Load and preprocess query image
    query_tensor = load_and_preprocess_image(query_image_path)
    if query_tensor is None:
        return

    # Generate query embedding
    query_embedding = generate_clip_embedding(query_tensor)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Display results
    if results and "ids" in results and results["ids"]:
        print("\n🎯 Top Similar Images:")
        for i, image_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            print(f"{i+1}. {metadata['image_name']} (Category: {metadata['category']}) - Path: {metadata['image_path']}")
    else:
        print("❌ No similar images found.")

# Run Full Pipeline
process_and_store_images(DATASET_PATH)

