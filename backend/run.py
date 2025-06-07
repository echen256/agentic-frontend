import asyncio, json, os, time, argparse, shutil, base64
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from PIL import Image
import numpy as np
import imagehash
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

ITERATION_TOKEN = "_:"

async def compare(current_context, current_url, target_image, target_width, target_height, current_iteration):
    page = await current_context.new_page()
    await page.goto(f"file://{os.path.abspath(current_url)}")
    
    # Change this so that the page takes a screenshot with the passed dimensions instead
    await page.set_viewport_size({"width": target_width, "height": target_height})
    screenshot_path = f"./qa/{current_iteration}.png"
    await page.screenshot(path=screenshot_path)
    await page.close()
    
    # Compare the target image to the current context
    screenshot_image = Image.open(screenshot_path)
    score = calculate_image_hash_similarity(screenshot_image, target_image)
    return score, screenshot_image

def calculate_image_hash_similarity(img1, img2):
    hash_1 = imagehash.average_hash(img1)
    hash_2 = imagehash.average_hash(img2)
    # Convert hash difference to similarity score (0-1, where 1 is identical)
    max_hash_diff = len(hash_1.hash) ** 2  # Max possible difference
    hash_diff = hash_1 - hash_2
    similarity = 1 - (hash_diff / max_hash_diff)
    return similarity

async def segment():
    pass

async def modify(current, iteration_count, target_image_path, screenshot_image):
    if (iteration_count == 0):
        # Init the loop by making a copy of the start html file with the suffix ITERATION_TOKEN + iteration_count and saving it.
        base_path = Path(current)
        new_filename = f"{base_path.stem}{ITERATION_TOKEN}{iteration_count}{base_path.suffix}"
        new_path = base_path.parent / new_filename
        shutil.copy2(current, new_path)
        current = str(new_path)
    else:
        # create a new file by splitting the iteration token, remaking it with the new iteration count, and saving it
        current_path = Path(current)
        if ITERATION_TOKEN in current_path.stem:
            base_name = current_path.stem.split(ITERATION_TOKEN)[0]
            new_filename = f"{base_name}{ITERATION_TOKEN}{iteration_count}{current_path.suffix}"
            new_path = current_path.parent / new_filename
            shutil.copy2(current, new_path)
            current = str(new_path)
    
    # Read the current HTML file content
    with open(current, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Encode the target image as base64
    with open(target_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    base64_screenshot_image = base64.b64encode(screenshot_image.read()).decode('utf-8')
    # Query OpenAI to modify the HTML file
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use vision-capable model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HTML/CSS developer. Your task is to modify HTML files to better match a target visual design shown in the image. Return only the modified HTML code, no explanations."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Here is the current HTML code that needs to be modified:\n\n{html_content}\n\nPlease modify this 
                            HTML to make it visually match the target design shown in the image. 
                            Focus on layout, styling, colors, fonts, spacing, and all visual elements. 
                            Return only the complete modified HTML code. The first image is the target image, and the second image 
                            is the current state of the HTML"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        },{
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_screenshot_image}",
                                "detail": "high"
                            }
                        }

                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        modified_html = response.choices[0].message.content
        modified_html = modified_html.replace("```html", "").replace("```", "")
                
        # Write the modified HTML back to the file
        with open(current, 'w', encoding='utf-8') as file:
            file.write(modified_html)
        
        print(f"Modified HTML file: {current} using OpenAI with target image")
        
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        print(f"Continuing with unmodified file: {current}")
    
    return current


async def main(target, start):
    score = 0
    threshold = 0.9
    iteration_count = 0
    max_iteration_count = 10
    
    # Load the target file here and set target_image, target width and target height with the loaded image
    target_image = Image.open(target)
    target_width, target_height = target_image.size
    
    current = start
    
    # Create qa directory if it doesn't exist
    os.makedirs("./qa", exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # Run in headless mode
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--window-size=1920,1080',
                '--start-maximized',
                '--disable-features=IsolateOrigins,site-per-process', # Disable site isolation
                '--disable-web-security',  # Disable CORS and other security features
                '--disable-site-isolation-trials',
                '--no-sandbox',
                '--ignore-certificate-errors',
                '--ignore-certificate-errors-spki-list',
                '--enable-features=NetworkService,NetworkServiceInProcess'
            ]
        )
        
        # Default user agent and headers for both contexts
        context_options = {
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'viewport': {'width': 1920, 'height': 1080},
            'extra_http_headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                'Sec-Ch-Ua-Platform': '"macOS"',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1'
            },
            'bypass_csp': True,  # Bypass Content Security Policy
            'ignore_https_errors': True  # Ignore HTTPS errors
        }
        # Create two persistent contexts - one for control and one for experimental
        current_context = await browser.new_context(**context_options)

        while (iteration_count < max_iteration_count):
            score, screenshot_image = await compare(current_context, current, target_image, target_width, target_height, iteration_count)
            if (score > threshold):
                print("Successfully edited component to threshold")
                await browser.close()
                return
            current = await modify(current, iteration_count, target, screenshot_image)
            print('------------')
            print(score)
            iteration_count += 1
        print("Failed to reach goal by max iteration count")
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agentic Frontend Editor')
    parser.add_argument('--target', type=str, default='./test_case/1/target.png',
                      help='Path to target output)')
    parser.add_argument('--start', type=str, default='./test_case/1/start.html',
                      help='Path to starting input)')
    
    args = parser.parse_args()
    print("Starting edit loop...")
    asyncio.run(main(args.target, args.start))

