import streamlit as st
import requests
import os
import tempfile
from PIL import Image

# FastAPI endpoint
API_URL = "http://localhost:8000"

st.title("Text to image-search")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# File upload section
st.header("Upload Images")
uploaded_files = st.file_uploader(
    "Drag and drop images here",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    # Create temp directory if not exists
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    image_paths = []
    
    # Display uploaded images
    st.subheader("Uploaded Images:")
    cols = st.columns(3)  # Display in 3 columns
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)
        
        # Show thumbnails in grid
        with cols[idx % 3]:
            st.image(uploaded_file, caption=uploaded_file.name, width=150)
    
    if st.button("Process Images", type="primary"):
        with st.spinner("Processing images with Gemini AI..."):
            try:
                response = requests.post(
                    f"{API_URL}/process_images/",
                    json={"image_paths": image_paths}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Images processed successfully! Processed {result['processed_count']} images.")
                    st.session_state.processed = True
                    
                    # Show processed metadata
                    with st.expander("View Processed Metadata"):
                        for metadata in result['metadata']:
                            st.json(metadata)
                else:
                    st.error(f"Failed to process images: {response.text}")
                    st.session_state.processed = False
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Query section - only show if images are processed
if st.session_state.processed:
    st.header("Search Images")
    
    # Provide example queries
    st.subheader("Example searches:")
    example_queries = [
        "people giving thumbs up",
        "outdoor sunny weather",
        "multiple people in photo",
        "indoor setting",
        "people making peace signs"
    ]
    
    cols = st.columns(len(example_queries))
    for idx, example in enumerate(example_queries):
        if cols[idx].button(example, key=f"example_{idx}"):
            st.session_state.query = example
    
    # Text input for custom query
    query = st.text_input(
        "Or describe what you're looking for:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., 'people giving thumbs up', 'outdoor sunny photos', etc."
    )
    
    if query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/query_images/",
                    json={"query": query}
                )
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    
                    if not results:
                        st.warning("No images matched your search. Try different keywords!")
                    else:
                        st.success(f"Found {len(results)} matching images")
                        
                        # Display results in a nice grid
                        for i, result in enumerate(results):
                            st.subheader(f"Match {i+1}")
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Display the image
                                image_url = f"{API_URL}{result['image_url']}"
                                try:
                                    st.image(image_url, width=300)
                                except Exception as e:
                                    st.error(f"Could not load image: {e}")
                                    st.write(f"Image URL: {image_url}")
                            
                            with col2:
                                st.write("**Metadata:**")
                                metadata = result["metadata"]
                                
                                # Format metadata nicely
                                st.write(f"**People:** {metadata.get('number_of_people', 'Unknown')}")
                                st.write(f"**Hand Signs:** {metadata.get('sign_used', 'None')}")
                                st.write(f"**Setting:** {metadata.get('landscape_description', 'Unknown')}")
                                st.write(f"**Weather:** {metadata.get('weather', 'Unknown')}")
                                st.write(f"**Mood:** {metadata.get('mood', 'Unknown')}")
                                st.write(f"**Image:** {metadata.get('image_name', 'Unknown')}")
                                
                                # Show match content
                                with st.expander("View Search Match Details"):
                                    st.write("**Content that matched your search:**")
                                    st.write(result.get("content", "No content available"))
                            
                            st.divider()
                            
                else:
                    st.error(f"Search failed: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000")
            except Exception as e:
                st.error(f"Search error: {str(e)}")

# Status check
with st.sidebar:
    st.header("System Status")
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{API_URL}/status/")
            if response.status_code == 200:
                status = response.json()
                st.success("API is running")
                st.json(status)
            else:
                st.error("API not responding properly")
        except:
            st.error("Cannot connect to API")
    
    st.info("Make sure your FastAPI server is running with: `python main.py`")