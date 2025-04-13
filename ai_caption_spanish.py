import os
import base64
import json
import time
import uuid
import pandas as pd
import cairosvg
from openai import OpenAI

# Setup
IMAGE_FOLDER = "files/test_set/"
BATCH_INPUT_FILE = "batchinput.jsonl"
DEBUG_PREVIEW_FILE = "debug_batch_preview.jsonl"
OUTPUT_CSV = "ai_caption_results_batch.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
BATCH_COMPLETION_WINDOW = "24h"

client = OpenAI(api_key=OPENAI_API_KEY)

# Fallback detection (Spanish)
def is_fallback_caption(caption: str) -> bool:
    if not caption:
        return True

    fallback_phrases = [
        "lo siento", 
        "no puedo ver la imagen", 
        "no puedo", 
        "descríbela para que pueda ayudarte", 
        "necesito la imagen o que me la describas", 
        "no se proporcionó imagen", 
        "no tengo acceso a la imagen",
        "soy un modelo de lenguaje de ia y no puedo ver imágenes",
        "no se pudo generar una descripción válida.",
        "claro, aquí tienes una descripción adecuada"
    ]

    caption_lower = caption.lower()
    return any(phrase in caption_lower for phrase in [p.lower() for p in fallback_phrases])

# Mojibake fixer
def fix_mojibake_characters(caption):
    replacements = {
        "√±": "ñ",
        "√©": "é",
        "√°": "á",
        "√≠": "í",
        "√≥": "ó",
        "√∫": "ú",
        "√º": "ú",
        "√ü": "ü",
        "√Ñ": "Ñ",
        "√â": "à",
        "√Æ": "Á",
        "√ã": "ã",
        "√¢": "â",
        "√ª": "ê",
        "√∞": "í",
        "√®": "É"
    }
    for bad, good in replacements.items():
        caption = caption.replace(bad, good)
    return caption

# Create batch input file
def create_batch_input():
    requests = []
    failed_files = []

    for svg_file in os.listdir(IMAGE_FOLDER):
        if not svg_file.endswith(".svg"):
            continue

        svg_path = os.path.join(IMAGE_FOLDER, svg_file)
        png_path = svg_path.replace(".svg", ".png")

        try:
            cairosvg.svg2png(url=svg_path, write_to=png_path)

            if not os.path.exists(png_path) or os.path.getsize(png_path) < 1000:
                raise Exception("Generated PNG is missing or too small.")

            with open(png_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            data_url = f"data:image/png;base64,{encoded}"
            custom_id = f"{os.path.splitext(svg_file)[0]}-{uuid.uuid4().hex[:8]}"

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
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
                                        "url": data_url
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 50
                }
            }

            try:
                test_json = json.dumps(request)
                json.loads(test_json)
                requests.append(request)
            except Exception as e:
                print(f"Invalid request JSON for {custom_id}: {e}")
                failed_files.append(svg_file)

            os.remove(png_path)

        except Exception as e:
            print(f"⚠️ Skipping file {svg_file} due to error: {e}")
            failed_files.append(svg_file)
            if os.path.exists(png_path):
                os.remove(png_path)

    with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    with open(DEBUG_PREVIEW_FILE, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    print(f"Batch input file created with {len(requests)} valid requests.")
    if failed_files:
        print(f"Skipped {len(failed_files)} files due to errors.")
        print("Check those SVGs manually if needed.")

# Upload + submit
def submit_batch():
    uploaded = client.files.create(
        file=open(BATCH_INPUT_FILE, "rb"),
        purpose="batch"
    )
    file_id = uploaded.id
    print(f"Uploaded batch file: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW
    )
    print(f"Batch submitted, Batch ID: {batch.id}")
    return batch.id

# Monitor
def wait_for_completion(batch_id):
    print("⏳ Waiting for batch to complete...")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Status: {batch.status}")
        if batch.status in ["completed", "failed", "expired", "cancelled"]:
            return batch
        time.sleep(30)

# Process results
def download_results(batch):
    if batch.status != "completed":
        print(f"Batch ended with status: {batch.status}")
        if batch.error_file_id:
            print("Downloading error file...")
            error_file = client.files.content(batch.error_file_id)
            with open("batch_errors.jsonl", "w", encoding="utf-8") as f:
                f.write(error_file.text)
            print("Error file saved to batch_errors.jsonl")
        else:
            print("No error file returned.")
        return

    output_file_id = batch.output_file_id
    result_file = client.files.content(output_file_id)
    lines = result_file.text.strip().split("\n")

    rows = []
    for line in lines:
        entry = json.loads(line)
        custom_id = entry.get("custom_id")
        caption = entry["response"]["body"]["choices"][0]["message"]["content"]

        caption = fix_mojibake_characters(caption)

        if is_fallback_caption(caption):
            caption = "[FALLBACK] " + caption

        rows.append({"Image Name": custom_id, "GPT-4 Vision Caption": caption})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {os.path.abspath(OUTPUT_CSV)}")

# Run it all
def main():
    create_batch_input()
    batch_id = submit_batch()
    batch = wait_for_completion(batch_id)
    download_results(batch)

if __name__ == "__main__":
    main()