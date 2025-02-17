import streamlit as st
import open_clip
import torch
import os
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS
from PIL import Image
import faiss
import requests
from io import BytesIO
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Initialize Vector Database (ChromaDB)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="image_search")

# Faiss Index for Deduplication
d = 512  # CLIP ViT-B/32 output dimension
faiss_index = faiss.IndexFlatL2(d)

# Initialize BLIP Model and Processor for Image Captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    # Preprocess the image and generate the caption using BLIP model
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def get_clip_embedding(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        return clip_model.encode_image(image).cpu().numpy()

def get_text_embedding(text):
    text_tokens = tokenizer([text])
    with torch.no_grad():
        return clip_model.encode_text(text_tokens).cpu().numpy()

def is_duplicate(embedding, threshold=0.99):
    if faiss_index.ntotal > 0:
        distances, _ = faiss_index.search(embedding.reshape(1, -1), 1)
        if distances[0][0] < (1 - threshold):
            return True
    return False

def add_image_to_db(image_name, image_path, embedding):
    if not is_duplicate(embedding):
        collection.add(ids=[image_name], embeddings=[embedding.tolist()], metadatas=[{"image_name": image_name, "image_path": image_path}])
        faiss_index.add(np.array([embedding], dtype=np.float32))
        return True
    return False

def retrieve_similar_images(input_embedding, top_k=5, similarity_threshold=0.25):
    results = collection.query(
        query_embeddings=[input_embedding.tolist()],
        n_results=top_k
    )

    if results and "metadatas" in results and results["metadatas"]:
        filtered_results = []
        for i, metadata in enumerate(results["metadatas"][0]):
            db_entry = collection.get(ids=[metadata["image_name"]], include=["embeddings"])
            if db_entry and "embeddings" in db_entry:
                db_embedding = np.array(db_entry["embeddings"][0], dtype=np.float32)
                similarity = cosine_similarity(db_embedding, input_embedding)
                
                if similarity >= similarity_threshold:
                    filtered_results.append(metadata)
        
        return filtered_results
    return []

def search_web_images(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)
    return [(res["image"], res.get("title", "No description available")) for res in results]

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_image_from_path_or_url(path_or_url):
    try:
        # Check if it's a URL (starts with http:// or https://)
        if path_or_url.startswith(('http://', 'https://')):
            response = requests.get(path_or_url, timeout=5)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Try to open as local file
            if os.path.exists(path_or_url):
                return Image.open(path_or_url).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {path_or_url}: {e}")
    return None

def display_image_result(path_or_url, source, score, caption):
    try:
        if path_or_url.startswith(('http://', 'https://')):
            # For web URLs, use st.image directly
            st.image(path_or_url, caption=f"Source: {source} (Similarity: {score:.2f})\n📝 {caption}", width=200)
        else:
            # For local files, open and display
            if os.path.exists(path_or_url):
                img = Image.open(path_or_url)
                st.image(img, caption=f"Source: {source} (Similarity: {score:.2f})\n📝 {caption}", width=200)
            else:
                st.error(f"Could not load image: {path_or_url}")
    except Exception as e:
        st.error(f"Error displaying image {path_or_url}: {e}")

def rank_results(collection, db_results, web_results, input_embedding, show_all):
    all_results = []
    
    # Process database results
    for item in db_results:
        image_name = item["image_name"]
        db_entry = collection.get(ids=[image_name], include=["embeddings", "metadatas"])
        if db_entry:
            db_embedding = db_entry["embeddings"][0]
            image_path = db_entry.get("metadatas", [{}])[0].get("image_path")
            if image_path:
                similarity = cosine_similarity(db_embedding, input_embedding)
                # Load and generate caption for database image
                db_image = get_image_from_path_or_url(image_path)  # Changed from get_image_from_url
                if db_image:
                    caption = generate_caption(db_image)
                    all_results.append((image_path, "Database", similarity, caption))
    
    # Process web results
    for img_url, original_caption in web_results:
        try:
            web_image = get_image_from_path_or_url(img_url)  # Changed from get_image_from_url
            if web_image:
                web_embedding = get_clip_embedding(web_image)[0]
                similarity = cosine_similarity(web_embedding, input_embedding)
                if not is_duplicate(web_embedding):
                    # Generate caption for web image
                    caption = generate_caption(web_image)
                    all_results.append((img_url, "Web Search", similarity, caption))
        except Exception as e:
            print(f"Could not process web image: {img_url} - {e}")
    
    ranked_results = sorted(all_results, key=lambda x: x[2], reverse=True)
    return ranked_results if show_all else ranked_results[:5]

# Streamlit UI
st.title("🔍 Image Search with AI Captioning")
option = st.radio("Select Search Mode:", ["Search by Image", "Search by Text", "Search by Image + Text"])
show_all = st.checkbox("Show all results (instead of top 5)")

if option == "Search by Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"], key="image_search_upload")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Processing image and generating caption..."):
            input_embedding = get_clip_embedding(image)[0]
            uploaded_caption = generate_caption(image)
            st.write(f"📝 Caption for uploaded image: {uploaded_caption}")

            db_results = retrieve_similar_images(input_embedding)
            web_results = search_web_images(uploaded_caption)  # Use generated caption for web search
            final_results = rank_results(collection, db_results, web_results, input_embedding, show_all)
            
            st.subheader(f"🔍 Showing {'All ' if show_all else 'Top 5'} Results")
            for img_url, source, score, caption in final_results:
                display_image_result(img_url, source, score, caption)

elif option == "Search by Text":
    query = st.text_input("Enter Search Query:")
    if query:
        with st.spinner("Searching and generating captions..."):
            input_embedding = get_text_embedding(query)[0]
            db_results = retrieve_similar_images(input_embedding)
            web_results = search_web_images(query)
            final_results = rank_results(collection, db_results, web_results, input_embedding, show_all)
            
            st.subheader(f"🔍 Showing {'All ' if show_all else 'Top 5'} Results")
            for img_url, source, score, caption in final_results:
                display_image_result(img_url, source, score, caption)

elif option == "Search by Image + Text":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"], key="image_text_search_upload")
    query = st.text_input("Enter Search Query:")
    if uploaded_file and query:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Processing inputs and generating captions..."):
            img_embedding = get_clip_embedding(image)[0]
            txt_embedding = get_text_embedding(query)[0]
            combined_embedding = (img_embedding + txt_embedding) / 2
            
            uploaded_caption = generate_caption(image)
            st.write(f"📝 Caption for uploaded image: {uploaded_caption}")
            
            db_results = retrieve_similar_images(combined_embedding)
            web_results = search_web_images(f"{query} {uploaded_caption}")  # Combine text with image caption
            final_results = rank_results(collection, db_results, web_results, combined_embedding, show_all)
            
            st.subheader(f"🔍 Showing {'All ' if show_all else 'Top 5'} Results")
            for img_url, source, score, caption in final_results:
                display_image_result(img_url, source, score, caption)
