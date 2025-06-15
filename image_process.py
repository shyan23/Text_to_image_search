import base64
import json
import os
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image
import shutil
import time

class ImageMetadataProcessor:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.metadata_store = []
    
    def save_to_public(self, image_path: str) -> str:
        """Copy image to public folder and return new path"""
        os.makedirs("public", exist_ok=True)
        filename = os.path.basename(image_path)
        public_path = os.path.join("public", filename)
        
        # Handle duplicate filenames
        counter = 1
        while os.path.exists(public_path):
            name, ext = os.path.splitext(filename)
            public_path = os.path.join("public", f"{name}_{counter}{ext}")
            counter += 1
        
        shutil.copy2(image_path, public_path)
        return public_path
    
    def get_image_description(self, image_path: str) -> str:
        """Get description of image from Gemini 1.5 Flash"""
        try:
            print(f"Processing image: {image_path}")
            
            # Try different approaches based on what works with your version
            try:
                # Method 1: Direct PIL Image approach
                img = Image.open(image_path)
                
                prompt = """Analyze this image and provide:
1. Number of people in the image
2. Any hand signs visible (V-sign, thumbs-up, peace sign, etc.) or 'none'
3. Landscape/setting description (indoor/outdoor, beach, mountains, city, etc.)
4. Weather condition if visible (sunny, cloudy, rainy, etc.) or 'unknown'
5. Overall mood/atmosphere of the image

Be concise and factual."""

                response = self.model.generate_content([prompt, img])
                return response.text
                
            except Exception as e1:
                print(f"Method 1 failed: {e1}")
                
                # Method 2: File upload approach
                try:
                    uploaded_file = genai.upload_file(image_path)
                    # Wait a moment for processing
                    time.sleep(1)
                    
                    response = self.model.generate_content([
                        "Analyze this image and describe: 1) Number of people 2) Hand signs 3) Landscape 4) Weather 5) Mood",
                        uploaded_file
                    ])
                    
                    # Clean up
                    genai.delete_file(uploaded_file.name)
                    return response.text
                    
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    
                    # Method 3: Base64 encoding (fallback)
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode()
                    
                    # This might not work with all versions, but worth trying
                    response = self.model.generate_content([
                        "Analyze this image: number of people, hand signs, landscape, weather, mood",
                        {"mime_type": "image/jpeg", "data": image_data}
                    ])
                    return response.text
            
        except Exception as e:
            print(f"All methods failed for {image_path}: {e}")
            return None
    
    def extract_metadata(self, description: str, image_path: str) -> Dict[str, Any]:
        """Extract structured metadata from description"""
        try:
            prompt = f"""Based on this image description, extract information as JSON:

Description: {description}

Return ONLY valid JSON in this exact format (no other text):
{{
    "sign_used": "describe any hand signs or 'none'",
    "number_of_people": 0,
    "landscape_description": "brief setting description",
    "weather": "weather condition or 'unknown'",
    "mood": "overall mood/atmosphere"
}}"""
            
            response = self.model.generate_content(prompt)
            json_str = response.text.strip()
            
            # Clean up the response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].strip()
            
            # Remove any extra text before/after JSON
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = json_str[start_idx:end_idx]
            
            print(f"Cleaned JSON string: {json_str}")
            
            metadata = json.loads(json_str)
            
            # Ensure all required fields exist
            required_fields = ["sign_used", "number_of_people", "landscape_description", "weather", "mood"]
            for field in required_fields:
                if field not in metadata:
                    metadata[field] = "unknown" if field != "number_of_people" else 0
            
            metadata['image_name'] = os.path.basename(image_path)
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response was: {response.text if 'response' in locals() else 'No response'}")
            
            # Fallback: create metadata from description text
            return self.create_fallback_metadata(description, image_path)
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return self.create_fallback_metadata(description, image_path)
    
    def create_fallback_metadata(self, description: str, image_path: str) -> Dict[str, Any]:
        """Create basic metadata when JSON extraction fails"""
        # Simple text analysis fallback
        desc_lower = description.lower()
        
        # Count people mentions
        people_count = 0
        people_words = ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl']
        for word in people_words:
            people_count += desc_lower.count(word)
        
        # Detect signs
        sign_used = "none"
        if any(sign in desc_lower for sign in ['v-sign', 'peace', 'thumbs up', 'victory']):
            sign_used = "hand signs detected"
        
        # Weather detection
        weather = "unknown"
        weather_words = ['sunny', 'cloudy', 'rainy', 'clear', 'overcast', 'bright']
        for word in weather_words:
            if word in desc_lower:
                weather = word
                break
        
        return {
            "sign_used": sign_used,
            "number_of_people": min(people_count, 10),  # Cap at reasonable number
            "landscape_description": description[:100] + "..." if len(description) > 100 else description,
            "weather": weather,
            "mood": "neutral",
            "image_name": os.path.basename(image_path)
        }
    
    def process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process images and return metadata"""
        self.metadata_store = []
        
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Image path does not exist: {path}")
                continue
                
            print(f"Processing image: {path}")
            public_path = self.save_to_public(path)
            description = self.get_image_description(public_path)
            
            if not description:
                print(f"Failed to get description for {path}")
                continue
                
            print(f"Description: {description}")
            metadata = self.extract_metadata(description, public_path)
            
            if metadata:
                self.metadata_store.append(metadata)
                print(f"Successfully processed: {metadata}")
            else:
                print(f"Failed to extract metadata for {path}")
                
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return self.metadata_store

class ImageRetriever:
    def __init__(self, metadata_store: List[Dict[str, Any]]):
        self.metadata_store = metadata_store
        self.vectorstore = None
        self.retriever = None
        self.docs = []
        
    def create_vector_store(self):
        """Create vector store from metadata"""
        if not self.metadata_store:
            raise ValueError("No metadata available to create vector store")
            
        # Create documents with rich content for better matching
        self.docs = []
        for entry in self.metadata_store:
            # Create comprehensive content that includes all searchable information
            content_parts = [
                f"Setting: {entry.get('landscape_description', '')}",
                f"Mood: {entry.get('mood', '')}",
                f"Weather: {entry.get('weather', '')}",
                f"People count: {entry.get('number_of_people', 0)}",
                f"Hand signs: {entry.get('sign_used', 'none')}"
            ]
            
            # Add descriptive terms based on metadata
            descriptive_terms = []
            
            # Add people-related terms
            people_count = entry.get('number_of_people', 0)
            if people_count == 0:
                descriptive_terms.append("no people")
            elif people_count == 1:
                descriptive_terms.append("single person")
            elif people_count > 1:
                descriptive_terms.append("multiple people")
                descriptive_terms.append("group")
            
            # Add sign-related terms
            sign_used = entry.get('sign_used', '').lower()
            if 'thumbs' in sign_used:
                descriptive_terms.extend(['thumbs up', 'positive gesture', 'approval'])
            elif 'peace' in sign_used or 'v-sign' in sign_used:
                descriptive_terms.extend(['peace sign', 'v sign', 'victory'])
            elif sign_used != 'none' and sign_used:
                descriptive_terms.append('hand gesture')
            
            # Add weather-related terms
            weather = entry.get('weather', '').lower()
            if weather in ['sunny', 'clear', 'bright']:
                descriptive_terms.extend(['sunny', 'bright', 'good weather', 'clear sky'])
            elif weather in ['cloudy', 'overcast']:
                descriptive_terms.extend(['cloudy', 'overcast', 'gray sky'])
            
            # Add setting-related terms
            landscape = entry.get('landscape_description', '').lower()
            if 'outdoor' in landscape or 'outside' in landscape:
                descriptive_terms.extend(['outdoor', 'outside', 'nature'])
            elif 'indoor' in landscape or 'inside' in landscape:
                descriptive_terms.extend(['indoor', 'inside'])
            
            if 'beach' in landscape:
                descriptive_terms.extend(['beach', 'sand', 'ocean', 'seaside'])
            elif 'mountain' in landscape:
                descriptive_terms.extend(['mountain', 'hills', 'elevation'])
            elif 'city' in landscape:
                descriptive_terms.extend(['city', 'urban', 'buildings'])
            
            # Combine all content
            full_content = ' '.join(content_parts + descriptive_terms)
            
            doc = Document(
                page_content=full_content,
                metadata={
                    "sign_used": entry.get("sign_used", "none"),
                    "number_of_people": entry.get("number_of_people", 0),
                    "landscape_description": entry.get("landscape_description", ""),
                    "weather": entry.get("weather", "unknown"),
                    "mood": entry.get("mood", "neutral"),
                    "image_name": entry.get("image_name", "")
                }
            )
            self.docs.append(doc)
        
        try:
            # Try Google embeddings first
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
            self.vectorstore = Chroma.from_documents(self.docs, embeddings)
            print("Vector store created successfully with Google embeddings")
        except Exception as e:
            print(f"Google embeddings failed: {e}")
            # Continue with simple search fallback
            self.vectorstore = None
        
        return self.vectorstore
    
    def simple_search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Enhanced search with better text matching"""
        if not self.docs:
            self.create_vector_store()
        
        results = []
        
        if self.vectorstore:
            try:
                # Use vector search if available
                docs = self.vectorstore.similarity_search(query, k=limit)
                results = [
                    {
                        "image_url": f"/public/{doc.metadata['image_name']}",
                        "metadata": doc.metadata,
                        "content": doc.page_content
                    }
                    for doc in docs
                ]
            except Exception as e:
                print(f"Vector search failed: {e}")
                # Fall back to simple search
        
        # If vector search failed or no results, use enhanced text matching
        if not results:
            results = self._enhanced_text_search(query, limit)
        
        return results
    
    def _enhanced_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Enhanced text matching with keyword expansion"""
        query_lower = query.lower()
        scored_docs = []
        
        # Expand query with synonyms
        expanded_terms = self._expand_query_terms(query_lower)
        
        for doc in self.docs:
            content = doc.page_content.lower()
            metadata = doc.metadata
            score = 0
            
            # Score based on exact matches
            for term in expanded_terms:
                if term in content:
                    score += content.count(term) * 2
            
            # Score based on metadata matching
            if 'thumbs' in query_lower and 'thumbs' in metadata.get('sign_used', '').lower():
                score += 10
            if 'peace' in query_lower and 'peace' in metadata.get('sign_used', '').lower():
                score += 10
            if 'people' in query_lower:
                people_count = metadata.get('number_of_people', 0)
                if people_count > 0:
                    score += people_count * 3
            if 'outdoor' in query_lower and 'outdoor' in metadata.get('landscape_description', '').lower():
                score += 5
            if 'indoor' in query_lower and 'indoor' in metadata.get('landscape_description', '').lower():
                score += 5
            if 'sunny' in query_lower and 'sunny' in metadata.get('weather', '').lower():
                score += 5
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and take top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_docs[:limit]:
            results.append({
                "image_url": f"/public/{doc.metadata['image_name']}",
                "metadata": doc.metadata,
                "content": doc.page_content,
                "score": score  # Include score for debugging
            })
        
        return results
    
    def _expand_query_terms(self, query: str) -> List[str]:
        """Expand query terms with synonyms for better matching"""
        terms = query.split()
        expanded = set(terms)  # Start with original terms
        
        # Add synonyms
        synonyms = {
            'thumbs': ['thumbs up', 'thumb', 'approval', 'positive'],
            'peace': ['peace sign', 'v sign', 'victory', 'v-sign'],
            'people': ['person', 'individuals', 'humans', 'group'],
            'outdoor': ['outside', 'nature', 'exterior'],
            'indoor': ['inside', 'interior'],
            'sunny': ['bright', 'clear', 'sunshine'],
            'happy': ['joyful', 'cheerful', 'positive'],
            'group': ['multiple', 'several', 'many']
        }
        
        for term in terms:
            if term in synonyms:
                expanded.update(synonyms[term])
        
        return list(expanded)
    
    def retrieve_images(self, query: str, limit: int = 3) -> List[Document]:
        """Backward compatibility method"""
        results = self.simple_search(query, limit)
        
        # Convert back to Document format for compatibility
        docs = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            docs.append(doc)
        
        return docs