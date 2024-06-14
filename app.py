import streamlit as st
import os
from dotenv import load_dotenv
import time
from time import perf_counter
from dotenv import load_dotenv
from libraries import ImageAnalysis, ImageModel, processing
from azure.cosmos import CosmosClient

# Load environment variables from .env file
load_dotenv(override=True)

cosmos_database = os.getenv("DATABASE") 
cosmos_container = os.getenv("CONTAINER_NAME")
cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
cosmos_key = os.getenv("COSMOS_KEY")
# Get the credentials from environment variables
subscription_key = os.getenv("VISION_KEY")
endpoint = os.getenv("VISION_ENDPOINT")

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")                           
api_key = os.getenv("AZURE_OPENAI_KEY")

if not subscription_key or not endpoint:
    st.error("Missing environment variables: VISION_ENDPOINT or VISION_KEY. Please set them in the .env file.")
    st.stop()



st.title("Albertson Image Keyword Extractor")
st.write("Enter an image URL to analyze its caption and tags.")

# Input URL from user
image_url = st.text_input("Image URL")

if st.button("Extract Metadata"):
    if image_url:
        try:
            st.image(image_url, caption="Input Image", use_column_width = True)
            analysis = ImageAnalysis(vision_endpoint=endpoint, vision_key=subscription_key)
            analysis.fit(image_path=image_url, url=True)
            ans = analysis.predict()
            ocr = processing(results=ans['OCR'])
            ocr_result = ocr.predict()

            if 'building' not in ans['items']:
                inference = ImageModel(azure_endpoint=azure_endpoint, api_key=api_key, deployment=deployment)
                inf = inference.predict(image_path=image_url, context=None)
            else:
                inf = {}
            del ans['OCR']

            if ans and inf is not None:
                status = {"status": 1}
                final = {**ans, **inf, **ocr_result, **status}
            else:
                status = {"status": 0}
                final = {**status}

            st.subheader("Results:")
            st.write(final)
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
    else:
        st.warning("Please enter a valid image URL.")


