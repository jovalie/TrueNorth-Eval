"""
Script to list available Google Generative AI models.
This will help identify which embedding models are available.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure the API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment variables")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 80)
print("Available Google Generative AI Models")
print("=" * 80)

# List all models
for model in genai.list_models():
    print(f"\nModel: {model.name}")
    print(f"  Display Name: {model.display_name}")
    print(f"  Description: {model.description}")
    print(f"  Supported Methods: {', '.join(model.supported_generation_methods)}")
    
print("\n" + "=" * 80)
print("Embedding-specific models:")
print("=" * 80)

# Filter for embedding models
for model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        print(f"\n✓ {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")

'''
Updated 3-28-2026
================================================================================
Available Google Generative AI Models
================================================================================

Model: models/gemini-2.5-flash
  Display Name: Gemini 2.5 Flash
  Description: Stable version of Gemini 2.5 Flash, our mid-size multimodal model that supports up to 1 million tokens, released in June of 2025.
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.5-pro
  Display Name: Gemini 2.5 Pro
  Description: Stable release (June 17th, 2025) of Gemini 2.5 Pro
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.0-flash
  Display Name: Gemini 2.0 Flash
  Description: Gemini 2.0 Flash
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.0-flash-001
  Display Name: Gemini 2.0 Flash 001
  Description: Stable version of Gemini 2.0 Flash, our fast and versatile multimodal model for scaling across diverse tasks, released in January of 2025.
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.0-flash-lite-001
  Display Name: Gemini 2.0 Flash-Lite 001
  Description: Stable version of Gemini 2.0 Flash-Lite
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.0-flash-lite
  Display Name: Gemini 2.0 Flash-Lite
  Description: Gemini 2.0 Flash-Lite
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.5-flash-preview-tts
  Display Name: Gemini 2.5 Flash Preview TTS
  Description: Gemini 2.5 Flash Preview TTS
  Supported Methods: countTokens, generateContent

Model: models/gemini-2.5-pro-preview-tts
  Display Name: Gemini 2.5 Pro Preview TTS
  Description: Gemini 2.5 Pro Preview TTS
  Supported Methods: countTokens, generateContent, batchGenerateContent

Model: models/gemma-3-1b-it
  Display Name: Gemma 3 1B
  Description: 
  Supported Methods: generateContent, countTokens

Model: models/gemma-3-4b-it
  Display Name: Gemma 3 4B
  Description: 
  Supported Methods: generateContent, countTokens

Model: models/gemma-3-12b-it
  Display Name: Gemma 3 12B
  Description: 
  Supported Methods: generateContent, countTokens

Model: models/gemma-3-27b-it
  Display Name: Gemma 3 27B
  Description: 
  Supported Methods: generateContent, countTokens

Model: models/gemma-3n-e4b-it
  Display Name: Gemma 3n E4B
  Description: 
  Supported Methods: generateContent, countTokens

Model: models/gemma-3n-e2b-it
  Display Name: Gemma 3n E2B
  Description: 
  Supported Methods: generateContent, countTokens

Model: models/gemini-flash-latest
  Display Name: Gemini Flash Latest
  Description: Latest release of Gemini Flash
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-flash-lite-latest
  Display Name: Gemini Flash-Lite Latest
  Description: Latest release of Gemini Flash-Lite
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-pro-latest
  Display Name: Gemini Pro Latest
  Description: Latest release of Gemini Pro
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.5-flash-lite
  Display Name: Gemini 2.5 Flash-Lite
  Description: Stable version of Gemini 2.5 Flash-Lite, released in July of 2025
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-2.5-flash-image
  Display Name: Nano Banana
  Description: Gemini 2.5 Flash Preview Image
  Supported Methods: generateContent, countTokens, batchGenerateContent

Model: models/gemini-2.5-flash-lite-preview-09-2025
  Display Name: Gemini 2.5 Flash-Lite Preview Sep 2025
  Description: Preview release (Septempber 25th, 2025) of Gemini 2.5 Flash-Lite
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-3-pro-preview
  Display Name: Gemini 3 Pro Preview
  Description: Gemini 3 Pro Preview
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-3-flash-preview
  Display Name: Gemini 3 Flash Preview
  Description: Gemini 3 Flash Preview
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-3.1-pro-preview
  Display Name: Gemini 3.1 Pro Preview
  Description: Gemini 3.1 Pro Preview
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-3.1-pro-preview-customtools
  Display Name: Gemini 3.1 Pro Preview Custom Tools
  Description: Gemini 3.1 Pro Preview optimized for custom tool usage
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-3.1-flash-lite-preview
  Display Name: Gemini 3.1 Flash Lite Preview
  Description: Gemini 3.1 Flash Lite Preview
  Supported Methods: generateContent, countTokens, createCachedContent, batchGenerateContent

Model: models/gemini-3-pro-image-preview
  Display Name: Nano Banana Pro
  Description: Gemini 3 Pro Image Preview
  Supported Methods: generateContent, countTokens, batchGenerateContent

Model: models/nano-banana-pro-preview
  Display Name: Nano Banana Pro
  Description: Gemini 3 Pro Image Preview
  Supported Methods: generateContent, countTokens, batchGenerateContent

Model: models/gemini-3.1-flash-image-preview
  Display Name: Nano Banana 2
  Description: Gemini 3.1 Flash Image Preview.
  Supported Methods: generateContent, countTokens, batchGenerateContent

Model: models/lyria-3-clip-preview
  Display Name: Lyria 3 Clip Preview
  Description: Lyria 3 30s model Preview
  Supported Methods: generateContent, countTokens

Model: models/lyria-3-pro-preview
  Display Name: Lyria 3 Pro Preview
  Description: Lyria 3 Pro Preview
  Supported Methods: generateContent, countTokens

Model: models/gemini-robotics-er-1.5-preview
  Display Name: Gemini Robotics-ER 1.5 Preview
  Description: Gemini Robotics-ER 1.5 Preview
  Supported Methods: generateContent, countTokens

Model: models/gemini-2.5-computer-use-preview-10-2025
  Display Name: Gemini 2.5 Computer Use Preview 10-2025
  Description: Gemini 2.5 Computer Use Preview 10-2025
  Supported Methods: generateContent, countTokens

Model: models/deep-research-pro-preview-12-2025
  Display Name: Deep Research Pro Preview (Dec-12-2025)
  Description: Preview release (December 12th, 2025) of Deep Research Pro
  Supported Methods: generateContent, countTokens

Model: models/gemini-embedding-001
  Display Name: Gemini Embedding 001
  Description: Obtain a distributed representation of a text.
  Supported Methods: embedContent, countTextTokens, countTokens, asyncBatchEmbedContent

Model: models/gemini-embedding-2-preview
  Display Name: Gemini Embedding 2 Preview
  Description: Obtain a distributed representation of multimodal content.
  Supported Methods: embedContent, countTextTokens, countTokens, asyncBatchEmbedContent

Model: models/aqa
  Display Name: Model that performs Attributed Question Answering.
  Description: Model trained to return answers to questions that are grounded in provided sources, along with estimating answerable probability.
  Supported Methods: generateAnswer

Model: models/imagen-4.0-generate-001
  Display Name: Imagen 4
  Description: Vertex served Imagen 4.0 model
  Supported Methods: predict

Model: models/imagen-4.0-ultra-generate-001
  Display Name: Imagen 4 Ultra
  Description: Vertex served Imagen 4.0 ultra model
  Supported Methods: predict

Model: models/imagen-4.0-fast-generate-001
  Display Name: Imagen 4 Fast
  Description: Vertex served Imagen 4.0 Fast model
  Supported Methods: predict

Model: models/veo-2.0-generate-001
  Display Name: Veo 2
  Description: Vertex served Veo 2 model. Access to this model requires billing to be enabled on the associated Google Cloud Platform account. Please visit https://console.cloud.google.com/billing to enable it.
  Supported Methods: predictLongRunning

Model: models/veo-3.0-generate-001
  Display Name: Veo 3
  Description: Veo 3
  Supported Methods: predictLongRunning

Model: models/veo-3.0-fast-generate-001
  Display Name: Veo 3 fast
  Description: Veo 3 fast
  Supported Methods: predictLongRunning

Model: models/veo-3.1-generate-preview
  Display Name: Veo 3.1
  Description: Veo 3.1
  Supported Methods: predictLongRunning

Model: models/veo-3.1-fast-generate-preview
  Display Name: Veo 3.1 fast
  Description: Veo 3.1 fast
  Supported Methods: predictLongRunning

Model: models/gemini-2.5-flash-native-audio-latest
  Display Name: Gemini 2.5 Flash Native Audio Latest
  Description: Latest release of Gemini 2.5 Flash Native Audio
  Supported Methods: countTokens, bidiGenerateContent

Model: models/gemini-2.5-flash-native-audio-preview-09-2025
  Display Name: Gemini 2.5 Flash Native Audio Preview 09-2025
  Description: Gemini 2.5 Flash Native Audio Preview 09-2025
  Supported Methods: countTokens, bidiGenerateContent

Model: models/gemini-2.5-flash-native-audio-preview-12-2025
  Display Name: Gemini 2.5 Flash Native Audio Preview 12-2025
  Description: Gemini 2.5 Flash Native Audio Preview 12-2025
  Supported Methods: countTokens, bidiGenerateContent

Model: models/gemini-3.1-flash-live-preview
  Display Name: Gemini 3.1 Flash Live Preview
  Description: Gemini 3.1 Flash Live Preview
  Supported Methods: bidiGenerateContent

================================================================================
Embedding-specific models:
================================================================================

✓ models/gemini-embedding-001
  Display Name: Gemini Embedding 001
  Description: Obtain a distributed representation of a text.

✓ models/gemini-embedding-2-preview
  Display Name: Gemini Embedding 2 Preview
  Description: Obtain a distributed representation of multimodal content.
'''