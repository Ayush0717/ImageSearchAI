 ImageSearchAI

Image Search AI is an intelligent image search application that allows users to search for images based on text queries. The system uses the DuckDuckGo API to fetch images from the web and then applies Transformers-based models to generate captions for the images. These captions are stored in a Vector Database(Chroma DB) for efficient searching and retrieval based on semantic similarity.

## Features
- **Image Search**: Fetches images based on user input text from DuckDuckGo.
- **Caption Generation**: Uses local transformers to generate descriptive captions for the images.
- **Vector Database**: Stores image captions in a Chroma DB, enabling fast and relevant image retrieval.
- **Similarity Search**: Allows users to query the system and retrieve images most semantically similar to their search.

## Installation

## Prerequisites
Ensure you have Python 3.7+ installed. You will also need an internet connection to access the DuckDuckGo API for image search.

## Running

**Clone the repository**:
  ```bash
   git clone https://github.com/Ayush0717/ImageSearchAI.git
   cd ImageSearchAI
   ```
Create a virtual environment (optional but recommended):


**Usage**

```bash
pip install -r requirements.txt

python db.py
streamlit run main.py


```
## Folder Structure

**dataset/**: Contains any dataset or sample images used.
**chroma_db/**: Stores the Chroma vector database. Running the vector database in Persistent Client Mode 
**main.py**: Main script that runs the application.
**db.py**: Script that handles image embedding and storing in the vector database.
**test.py**: A test script to verify the search functionality.


## Technologies Used

DuckDuckGo API for image search.
Transformers (Hugging Face) for caption generation.
Chroma DB for storing and searching captions in a vector database.
Python for scripting and automation.
