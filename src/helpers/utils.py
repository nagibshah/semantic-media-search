import os
import io
import boto3
from pathlib import Path
import pandas as pd
import numpy as np
import json
import base64

from pathlib import Path
from PIL import Image
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

from io import BytesIO
from typing import List, Union 
from sagemaker.s3 import S3Downloader as s3down

session = boto3.session.Session()
region = session.region_name

# Define bedrock client
bedrock_client = boto3.client(
    "bedrock-runtime", 
    region, 
    endpoint_url=f"https://bedrock-runtime.{region}.amazonaws.com"
)


# Bedrock models
# Select Amazon titan-embed-image-v1 as Embedding model for multimodal indexing
multimodal_embed_model = f'amazon.titan-embed-image-v1'

"""
Function to generate Embeddings from image or text
"""
def get_titan_multimodal_embedding(
    image_path:str=None,  # maximum 2048 x 2048 pixels
    description:str=None, # English only and max input tokens 128
    dimension:int=1024,   # 1,024 (default), 384, 256
    model_id:str=multimodal_embed_model
):
    payload_body = {}
    embedding_config = {
        "embeddingConfig": { 
             "outputEmbeddingLength": dimension
         }
    }
    # You can specify either text or image or both
    if image_path:
        with open(image_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')
        payload_body["inputImage"] = input_image
    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"
    # print("\n".join(payload_body.keys()))

    response = bedrock_client.invoke_model(
        body=json.dumps({**payload_body, **embedding_config}), 
        modelId=model_id,
        accept="application/json", 
        contentType="application/json"
    )

    return json.loads(response.get("body").read())

"""
function to generate an image using titan image generator model 
"""
def generate_titan_image(prompt, neg_prompt, seed=1, number_of_images=1):
    # Create payload
    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,                    # Required
                "negativeText": neg_prompt   # Optional
            },
            "imageGenerationConfig": {
                "numberOfImages": number_of_images,   # Range: 1 to 5 
                "quality": "premium",  # Options: standard or premium
                "height": 512,        # Supported height list in the docs 
                "width": 512,         # Supported width list in the docs
                "cfgScale": 7.5,       # Range: 1.0 (exclusive) to 10.0
                "seed": seed             # Range: 0 to 214783647
            }
        }
    )
    # Make model request
    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-image-generator-v1",
        accept="application/json", 
        contentType="application/json"
    )
    # Process the image payload
    response_body = json.loads(response.get("body").read())
    img1_b64 = response_body["images"][0] # in bytes IO

    img = Image.open(io.BytesIO(base64.decodebytes(bytes(img1_b64, "utf-8"))))
    
    return img

"""
function to generate description of an image
"""
def generate_img_desc(prompt,img_path, max_tokens=512,temp=0.5,top_p=0.9):

    modelid = 'anthropic.claude-3-sonnet-20240229-v1:0'
    
    # Read reference image from file and encode as base64 strings.
    img_b64 = ""
    with open(img_path, "rb") as image_file:
        img_b64 = base64.b64encode(image_file.read()).decode('utf8')
        
    message_mm = [
        { "role": "user",
          "content": [
          {"type": "image","source": { "type": "base64","media_type":"image/jpeg","data": img_b64}},
          {"type": "text","text": prompt}
          ]
        }
    ]

    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": message_mm,
            "temperature": temp,
            "top_p": top_p
        }  
    )  
    
    response = bedrock_client.invoke_model(body=body, modelId=modelid)
    response_body = json.loads(response.get('body').read())
    response_text = response_body["content"][0]["text"]    

    return response_text

""" 
Function to plot heatmap from embeddings
"""

def plot_similarity_heatmap(embeddings_a, embeddings_b):
    inner_product = np.inner(embeddings_a, embeddings_b)
    sns.set(font_scale=1.1)
    graph = sns.heatmap(
        inner_product,
        vmin=np.min(inner_product),
        vmax=1,
        cmap="OrRd",
    )

""" 
Function to fetch the image based on image id from dataset
"""
def get_image_from_item_id( item_id = "0", dataset = None, return_image=True):
 
    item_idx = dataset.query(f"item_id == {item_id}").index[0]
    img_path = dataset.iloc[item_idx].image_path
    
    if return_image:
        img = Image.open(img_path)
        return img, dataset.iloc[item_idx].item_desc
    else:
        return img_path, dataset.iloc[item_idx].item_desc
    print(item_idx,img_path)


""" 
Function to fetch the image based on image id from S3 bucket
"""
    
def get_image_from_item_id(image_id = "B0896LJNLH", dataset = None, image_path = None,  return_image=True):

    item_idx = dataset.query(f"image_id == '{image_id}'").index[0]
    img_loc = dataset.iloc[item_idx].path

    if return_image:
        img = Image.open(img_loc)
        return img, dataset.iloc[item_idx].title
    else:
        return img_loc, dataset.iloc[item_idx].title

""" 
Function to display the images.
"""
def display_images(
    images: [Image], 
    columns=3, width=10, height=10, max_images=15, 
    label_wrap_length=50, label_font_size=8):
 
    if not images:
        print("No images to display.")
        return 
 
    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]
 
    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):
 
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)
 
        if hasattr(image, 'name_and_score'):
            plt.title(image.name_and_score, fontsize=label_font_size); 
            



def display_results(res, dataset):
    images = []
    for hit in res["hits"]["hits"]:
        item_id_ = hit["_source"]["image_id"]
        image, item_name = get_image_from_item_id(image_id = item_id_, dataset = dataset,image_path = None)
        image.name_and_score = f'{hit["_score"]}:{item_name}'
        images.append(image)
    display_images(images)
    return 


