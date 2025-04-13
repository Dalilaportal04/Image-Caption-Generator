# Image-Caption-Generator
This project automates the process of generating short, accessible captions for a large set of SVG images using the power of OpenAI’s GPT-4 Vision model via the Batch API. It’s built for scale, precision, and speed — whether you’re working with hundreds or hundreds of thousands of images.

The tool takes each .svg image, converts it into a .png, encodes it as base64, and constructs a structured request that includes a clear, concise prompt asking GPT-4 to describe the object in the image using 1 to 6 simple words. This makes it ideal for generating alt-text for icons, math visuals, or educational graphics, especially for child-focused platforms and accessibility use cases.

Once all the requests are prepared, the tool uses OpenAI’s Batch API to upload and submit them for processing in one go. Unlike synchronous API calls, the batch approach is faster and more efficient for large datasets, eliminating the need to handle rate limits or manual retries.

The script also monitors the batch’s progress in real time and automatically downloads the results once completed. It intelligently identifies fallback responses (e.g., “I can’t see the image”) using a list of known phrases in both English and Spanish, marking them clearly for filtering or post-processing. This helps ensure your dataset remains clean and reliable.

To further improve output quality, the script includes a utility to fix mojibake — those weird character glitches that sometimes happen with special characters like ñ, é, or á. It replaces them with the correct characters to ensure proper formatting in the final CSV output.

By the end of the process, you’ll have a CSV file containing each image ID and its corresponding caption — perfect for use in web apps, accessibility tools, datasets, or content labeling pipelines.

Requirements:
You’ll need an OpenAI API key with access to the GPT-4 Vision model. The script uses Python 3 with the following libraries: openai, cairosvg, pandas, uuid, base64, and os. Install dependencies with pip install -r requirements.txt.
