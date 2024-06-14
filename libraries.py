import base64
import cv2
import mlflow.pyfunc
from openai import AzureOpenAI
import json
import re
import requests
from PIL import Image
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from typing import Any, Callable, Dict, List, Optional

import faiss  # type: ignore
import numpy as np
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


class SemanticRetriever(BaseEstimator):
    """
    Class to manage a vector store with Faiss indexing, compatible with scikit-learn API.
    """

    def __init__(self, embedding_size: int, embedding_function: Callable, df: pd.DataFrame):
        """
        Initialize the VectorStore.

        Args:
            embedding_size (int): The size of the embeddings.
            embedding_function (callable): Function to generate embeddings.
            df (DataFrame): Input as a dataframe
        """
        self.embedding_size = embedding_size
        self.embedding_function = embedding_function
        self.df = df
        self.index = faiss.IndexFlatIP(self.embedding_size)

        

    def add_data(self) -> None:
        """
        Add data to the vector store.
        """
        try:

            reviews_list = self.df.comments_processed.to_list()
            embeddings_list = self.df.embeddings.to_list()
            review_id_list = self.df.review_id.to_list()

            # Initialize vector store with FAISS and add embeddings
            self.vectorstore = FAISS(self.embedding_function, self.index, InMemoryDocstore(), {})
            self.vectorstore.add_embeddings(
                zip(reviews_list, embeddings_list),
                metadatas=[
                    {"source": "semantic_search", "review_id": review_id_list[i]} for i in range(len(review_id_list))
                ],
            )

        except Exception as e:
            print(f"Failed due to: {e}")

    def fit(self) -> None:
        """
        Fit the VectorStore.
        """
        self.add_data()
        return None

    def predict(self, query: str, n_results: int) -> Any:
        """
        Perform similarity search using Faiss retrieval.

        Args:
            query (str): Query string.
            n_results (int): Number of results to retrieve.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved documents.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Call fit method first.")

        # Retrieve documents similar to the query
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": n_results})
        faiss_results = faiss_retriever.get_relevant_documents(query)
        faiss_retrieved_dict = {i + 1: result.page_content for i, result in enumerate(faiss_results)}
        faiss_df = pd.DataFrame.from_dict(faiss_retrieved_dict, orient="index", columns=["page_content"])
        faiss_df.index.name = "rank"
        faiss_df.reset_index(inplace=True)
        return faiss_df

class LexicalRetriever(BaseEstimator):
    """
    Class to perform document retrieval using BM25 algorithm, compatible with scikit-learn API.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the BM25Retriever.

        Args:
            df (DataFrame): Input dataframe containing 'comments_processed' and 'review_id' columns.
        """

        self.df = df
        self.bm25_retriever = None

    def initialize_retriever(self) -> None:
        """
        Initialize the BM25Retriever.
        """
        try:
            reviews = self.df.comments_processed.to_list()
            review_ids = self.df.review_id.to_list()
            self.bm25_retriever = BM25Retriever.from_texts(  
                reviews,
                metadatas=[{"source": 1, "review_id": review_ids[i]} for i in range(len(review_ids))],
            )
        except Exception as e:
            raise ValueError(f"Error initializing BM25Retriever: {e}")

    def fit(self) -> None:
        """
        Fit the BM25Retrieval.
        """
        self.initialize_retriever()
        return None

    def predict(self, query: str, k: int = 30) -> pd.DataFrame:
        """
        Retrieve documents using BM25 retrieval.

        Args:
            query (str): Query string.
            k (int): Number of results to retrieve.

        Returns:
            pd.DataFrame: DataFrame containing the retrieved documents.
        """
        if self.bm25_retriever is None:
            raise ValueError("BM25 Retriever is not initialized. Call fit method first.")

        self.bm25_retriever.k = k
        bm25_results = self.bm25_retriever.get_relevant_documents(query)
        bm25_retrieved_dict = {i + 1: result.page_content for i, result in enumerate(bm25_results)}
        bm25_df = pd.DataFrame.from_dict(bm25_retrieved_dict, orient="index", columns=["page_content"])
        bm25_df.index.name = "rank"
        bm25_df.reset_index(inplace=True)
        return bm25_df

class RankMerger(BaseEstimator):
    """
    Class to merge rankings from multiple sources and calculate relevance scores, compatible with scikit-learn API.
    """

    def __init__(
        self,
        semantic_w: float = 0.0,
        lexical_w: float = 0.0,
        semantic_df: pd.DataFrame = pd.DataFrame(),
        lexical_df: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Initialize the RankMerger.

        Args:
            semantic_w (float): Weight for semantic retriever.
            lexical_w (float): Weight for lexical retriever.
            semantic_df (pd.DataFrame): Semantic DataFrame containing 'rank' and 'page_content' columns.
            lexical_df (pd.DataFrame): Lexical DataFrame containing 'rank' and 'page_content' columns.

        """

        # Checking sum of weights
        total_weight = semantic_w + lexical_w
        if total_weight != 1.0:
            raise ValueError("Sum of all weights must be equal to 1.0.")

        self.semantic_w = semantic_w
        self.lexical_w = lexical_w
        self.semantic_df = semantic_df
        self.lexical_df = lexical_df

        self.merged_dict: dict[Any, Any] = dict()
        self.final_df: Optional[pd.DataFrame] = None


    def merge_and_calculate_reciprocal(self) -> None:
        """
        Merge rankings from multiple sources and calculate reciprocal ranks.
        """
        dataframes = [self.semantic_df, self.lexical_df, self.length_df, self.recency_df, self.attribute_df]
        weights = [self.semantic_w, self.lexical_w, self.length_w, self.recency_w, self.attribute_w]

        self.merged_dict = dict()

        # Iterating over each DataFrame and weight
        for df, weight in zip(dataframes, weights):
            # Validate input data against the defined schema
            try:
                if not df.empty:  # Checking if DataFrame is not empty

                    df_dict = df.set_index("page_content")["rank"].to_dict()

                    # Merge the dictionaries
                    for key, value in df_dict.items():
                        if key not in self.merged_dict:
                            self.merged_dict[key] = list()
                        self.merged_dict[key].append((value, weight))

            except pa.errors.SchemaError as e:
                raise ValueError(f"Input dataframe validation failed: {e}")

        # Storing the ranks
        for key, values in self.merged_dict.items():
            self.merged_dict[key] = [value for value in sorted(values)]

    def rank_fusion(self) -> dict:
        """
        Calculate relevance scores.
        """
        if self.merged_dict is None:
            raise ValueError("Merge the rankings first using merge_and_calculate_reciprocal method.")

        result = {}
        for k, v in self.merged_dict.items():
            score = 0.0
            for i, j in v:
                score += j * 1 / (60 + i)
            result[k] = score
        return result

    def fit(self) -> None:
        """
        Fit the RankMerger.
        """
        self.merge_and_calculate_reciprocal()

    def predict(self) -> pd.DataFrame:
        """
        Generate the final output dataframe.

        Returns:
            pd.DataFrame: DataFrame containing the final merged and normalized relevance scores.
        """
        if self.final_df is None:
            final_scores = self.rank_fusion()
            self.final_df = pd.DataFrame.from_dict(final_scores, orient="index").reset_index()
            self.final_df.columns = pd.Index(["page_content", "relevance_score"])
            self.final_df = self.final_df.sort_values(by="relevance_score", ascending=False)
            self.final_df = self.final_df[["page_content", "relevance_score"]]
            return self.final_df


