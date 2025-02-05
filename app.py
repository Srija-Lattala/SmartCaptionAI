import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from typing import Tuple, Optional
import openai
import os
import logging
from dotenv import load_dotenv, find_dotenv

# Try to find .env file
env_path = find_dotenv()
print(f"Found .env file at: {env_path}")

# Load the .env file
if load_dotenv(env_path):
    print("Successfully loaded .env file")
else:
    print("Failed to load .env file!")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"OpenAI version: {openai.__version__}")  # Should be 0.28.0


@st.cache_resource
def load_blip_models() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Load BLIP models with Streamlit caching."""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        logger.error(f"Error loading BLIP models: {str(e)}")
        raise


@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analyzer with Streamlit caching."""
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        logger.error(f"Error loading sentiment analyzer: {str(e)}")
        raise


class ImageCaptioningApp:
    def __init__(self):
        # Add debug prints
        print("Loading environment variables...")
        print(f"Available env vars: {list(os.environ.keys())}")
        
        self.openai_api_key = self._load_api_key()
        print(f"API key loaded: {'Yes' if self.openai_api_key else 'No'}")
        
        # Temporary direct setting for testing
        openai.api_key = self.openai_api_key
        print(f"OpenAI API key directly set to: {openai.api_key[:5]}...")  # Print first 5 chars for safety
        
        self.processor = None
        self.model = None
        self.sentiment_analyzer = None

    @staticmethod
    def _load_api_key() -> str:
        """Safely load OpenAI API key from environment variables."""
        # Try both environment variable names
        api_key = os.getenv("OPENAPIKEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: No API key found in environment variables!")
            print(f"Current working directory: {os.getcwd()}")
            raise ValueError("OpenAI API key not found in environment variables")
        return api_key

    def load_models(self) -> None:
        """Load all required models using cached functions."""
        try:
            self.processor, self.model = load_blip_models()
            self.sentiment_analyzer = load_sentiment_analyzer()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def generate_alt_text(self, image: Image.Image) -> Optional[str]:
        """Generate alt text for the given image."""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            alt_text = self.processor.decode(out[0], skip_special_tokens=True)
            return alt_text
        except Exception as e:
            logger.error(f"Error generating alt text: {str(e)}")
            return None

    def enhance_caption(self, alt_text: str) -> Optional[str]:
        """Enhance the caption using OpenAI's GPT model."""
        try:
            # For OpenAI v1.x
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative caption generator."},
                    {"role": "user", "content": self._create_enhancement_prompt(alt_text)}
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"Error enhancing caption: {str(e)}")
            return None

    @staticmethod
    def _create_enhancement_prompt(alt_text: str) -> str:
        """Create the prompt for caption enhancement."""
        return f"""
        Below is an alt text description of an image. 
        Please enhance it into a more engaging and creative caption while maintaining accuracy:

        Alt Text: {alt_text}
        """

    def analyze_sentiment(self, text: str) -> Optional[dict]:
        """Analyze the sentiment of the given text."""
        try:
            if not text:
                return None
            return self.sentiment_analyzer(text, truncation=True)[0]
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return None

    def run(self):
        """Run the Streamlit application."""
        # Add custom styling and layout
        st.set_page_config(
            page_title="Image Caption Generator",
            page_icon="🖼️",
            layout="wide"
        )

        # Create a cleaner header with custom styling
        st.markdown("""
            <h1 style='text-align: center; color: #2E86C1;'>
                🖼️ Image Caption Generator
            </h1>
            <p style='text-align: center; color: #666666;'>
                Upload an image to generate an engaging caption with sentiment analysis
            </p>
        """, unsafe_allow_html=True)

        # Add sidebar with information
        with st.sidebar:
            st.markdown("### About")
            st.info("""
                This app uses AI to:
                1. Generate alt text for images
                2. Create engaging captions
                3. Analyze sentiment
            """)

        # Load models in a collapsible section
        with st.expander("Model Status", expanded=False):
            try:
                self.load_models()
                st.success("✅ All models loaded successfully")
            except Exception as e:
                st.error(f"Failed to load models: {str(e)}")
                return

        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        # File uploader in the left column
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG"
            )

        if uploaded_file:
            try:
                # Display image in left column
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                # Results in right column
                with col2:
                    with st.container():
                        # Alt text generation
                        with st.status("🤖 Processing image...", expanded=True) as status:
                            st.write("Generating alt text...")
                            alt_text = self.generate_alt_text(image)
                            if alt_text:
                                st.success("Alt text generated!")
                                status.update(label="✨ Results", state="complete", expanded=True)
                            else:
                                st.error("Failed to generate alt text")
                                return

                        # Display results in tabs
                        tab1, tab2, tab3 = st.tabs(["Alt Text", "Enhanced Caption", "Sentiment"])
                        
                        with tab1:
                            st.markdown("### Generated Alt Text")
                            st.write(alt_text)

                        with tab2:
                            st.markdown("### Enhanced Caption")
                            enhanced_caption = self.enhance_caption(alt_text)
                            if enhanced_caption:
                                st.write(enhanced_caption)
                            else:
                                st.error("Failed to enhance caption")

                        with tab3:
                            st.markdown("### Sentiment Analysis")
                            sentiment = self.analyze_sentiment(enhanced_caption)
                            if sentiment:
                                col_s1, col_s2 = st.columns(2)
                                with col_s1:
                                    st.metric("Sentiment", sentiment['label'])
                                with col_s2:
                                    st.metric("Confidence", f"{sentiment['score']:.2%}")
                            else:
                                st.error("Failed to analyze sentiment")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    app = ImageCaptioningApp()
    app.run()
