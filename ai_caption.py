import os
import time
import base64
import pandas as pd
from PIL import Image
import cairosvg
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from nltk.translate.meteor_score import single_meteor_score
import nltk

# Download required resources for METEOR score
nltk.download("wordnet")
nltk.download("omw-1.4")

# Configuration
IMAGE_FOLDER = "files/test_set/"
OUTPUT_CSV = "ai_caption_results.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key")

client = OpenAI(api_key=OPENAI_API_KEY)

# Convert SVG file to PNG format
def convert_svg_to_png(svg_path):
    png_path = svg_path.replace(".svg", ".png")
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path)
        return png_path
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        return None

# Check if the PNG is valid and can be opened
def validate_png(image_path):
    try:
        with Image.open(image_path) as img:
            return img.format == "PNG"
    except Exception as e:
        print(f"Invalid PNG: {e}")
        return False

# Check if the model response is a fallback or non-visual response
def is_fallback_caption(caption: str) -> bool:
    if not caption:
        return True
    fallback_phrases = [
        "i'm sorry", "i cannot", "can't see the image", "I'm", "I am", "GPT-4",
        "could you describe", "unable to see", "i am not able", "I'm unable",
        "no image provided", "image not available", "i can't access", "Certainly!", "provide the image"
    ]
    return any(phrase in caption.lower() for phrase in fallback_phrases)

# Request a caption from GPT-4 Vision using a base64 image
def get_caption_gpt4(image_path, retries=2):
    if not validate_png(image_path):
        return "Invalid PNG"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the image in the context of a kid website for accessibility giving one word or max 6 words saying what the object is including numbers. Don't say emojis."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}",
                                    "detail": "low"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=50,
            )

            caption = response.choices[0].message.content.strip()
            if is_fallback_caption(caption):
                print(f"Fallback response (attempt {attempt + 1}): {caption}")
                time.sleep(1)
                continue

            return caption

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1)

    return "GPT-4 failed to describe this image after retries."

# Handle the full process for a single SVG image
def process_image(svg_file):
    svg_path = os.path.join(IMAGE_FOLDER, svg_file)
    png_path = convert_svg_to_png(svg_path)
    if not png_path:
        return None

    caption = get_caption_gpt4(png_path)
    os.remove(png_path)

    return {
        "Image Name": svg_file,
        "GPT-4 Vision Caption": caption
    }

# Run the full pipeline over all SVGs in the folder
def process_all_images():
    start = time.time()
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".svg")]

    print(f"Found {len(image_files)} SVG files.")

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_image, image_files))

    results = [r for r in results if r]
    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_CSV, index=False)

    total_time = time.time() - start
    avg_time = total_time / len(results) if results else 0
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average per image: {avg_time:.2f} seconds")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_all_images()
