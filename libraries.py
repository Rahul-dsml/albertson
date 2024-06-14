import base64

import mlflow.pyfunc
from openai import AzureOpenAI
import json
import re
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from typing import Any, Callable,  Optional

import faiss  # type: ignore
import pandas as pd
from sklearn.base import BaseEstimator
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

class ImageModel(mlflow.pyfunc.PythonModel):
    def __init__(self, azure_endpoint, api_key, deployment):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version="2024-02-01")
        

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    def call_model(self, image_path):
        #base64_image = self.encode_image_to_base64(image_path)
        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", 
                       "content": [{"type": "text", 
                                    "text": f"""Given an image, your task is to do multilabel classification on the basis of - `Theme`, `Department` and `Meal Type`.
                                    The categories for these tags are:
                                    Theme: ["St. Patrick's Day", "Easter", "Cinco de Mayo", "Mother's Day", "Memorial Day", "Graduation", "Father's Day", "4th of July", "Back to School", "Labor Day", “Oktoberfest", "Fall Football", "Halloween", "Thanksgiving", "Christmas", "Hanukkah",  "New Year's Eve", "Super Bowl", "Valentine's Day", "Mardi Gras",  "Lunar New Year", "Holiday"]
                                    Department: ["Deli", "Bakery", "Meat", "Seafood", "Produce", "Floral", "Center Store", "Dairy", "Front End", "Frozen", "Spirits", "Pharmacy”]
                                    Meal Type: ["Breakfast", "Lunch", "Dinner", "Dessert", "Appetizer", "Drinks"]

                                    You must return the output strictly in JSON format in curly brackets with keys as tags and values as list of labels.
                                    """},
                                   {"type": "image_url",
                                    "image_url": {"url": image_path, "detail": "high"}}]
                                        # {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}]
                            }],
        )
        return completion.choices[0].message.content
    
    def load_context(self, context):
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version="2024-02-01"
        )

    def predict(self, image_path, context):
        # image_path = model_input["image_path"]
        ans = self.call_model(image_path)
        start_index = ans.find('{')

        end_index = ans.rfind('}')

        json_string = ans[start_index:end_index+1]
        data = json.loads(json_string)
        return data
    
class ImageAnalysis(mlflow.pyfunc.PythonModel):
    
    def __init__(self, vision_key: str, vision_endpoint: str):
        
        self.client = ImageAnalysisClient(endpoint=vision_endpoint,
                                   credential=AzureKeyCredential(key=vision_key))
        
    def fit(self, image_path: str, url: bool = False):
        self.url = url
        if url is True:
            print("Using URL")
            # img = Image.open(requests.get(image_path, stream=True).raw)
            # Image_Representation=image_url
            self.result = self.client.analyze_from_url(image_url=image_path,
                                        visual_features=[VisualFeatures.CAPTION, 
                                                        VisualFeatures.TAGS, 
                                                        VisualFeatures.READ])
            return self.result
        else:
            print("using local")
            with open(image_path, "rb") as f:
                image_data = f.read()
            self.result = self.client.analyze(image_data=image_data,
                                        visual_features=[VisualFeatures.CAPTION, 
                                                        VisualFeatures.TAGS, 
                                                        VisualFeatures.READ])
            return self.result
    
    def predict(self):
        ocr=[]
        items=[]
        caption = self.result['captionResult']['text']
        if len(self.result.read.blocks) !=0:
            for line in self.result.read.blocks[0].lines:
                ocr.append(line.text)
        
        tags = self.result['tagsResult']['values']
        items = [i['name'] for i in tags if i['confidence'] >=0.85]
        answer = {"items": items,
                  "OCR": ocr,
                  "caption": caption}
        return answer

# Function to find matches and return the output as a dictionary
class processing:
    
    def __init__(self, results):
        self.results = results
        self.banners = ["ACME", "Albertsons", "Jewel-Osco", "Safeway", "Shaws", "Vons", "Thumb", "Randalls", "United", "Star Market", "Pavilions", "Haggen", "Market Street"]
        self.brands = ["O Organics", "Open Nature", "Signature Select", "Signature Care", "Lucerne", "Waterfront Bistro", "Primo Taglio", "Debi Lilly", "Value Corner", "Soleil", "Ready Meals"]
        self.dietary_preferences = ["VEGETARIAN", "PALEO FRIENDLY", "VEGAN", "CARB CONSCIOUS", "KETO FRIENDLY", "GLUTEN FREE", "NIGHTSHADE FREE", "EGG FREE", "SESAME FREE", "MUSTARD FREE", "SULFITE FREE", "DAIRY FREE", "SHELLFISH FREE", "TREE NUT FREE", "PEANUT FREE", "FISH FREE", "SOY FREE", "Kosher"]
        
    def normalize_strings(self, s):
        return re.sub(r'[\s\-]', '', s.lower())
    
    def find_matches(self, main_list, normalized_list):
        matched_list=[]
        for item in main_list:
            normalized_item = self.normalize_strings(item)
            if normalized_item in normalized_list:
                matched_list.append(item)
        return matched_list
    
    def predict(self):
        normalized_results = [self.normalize_strings(result) for result in self.results]

        banner_result = self.find_matches(self.banners, normalized_results)
        brand_result = self.find_matches(self.brands, normalized_results)
        diet_result = self.find_matches(self.dietary_preferences, normalized_results)

        return {
            "banners": banner_result,
            "brands": brand_result,
            "dietary_preferences": diet_result
    }

