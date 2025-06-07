import asyncio, json, os, time, argparse
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

ITERATION_TOKEN = "_:"

async def main(target, start):
    score = 0
    threshold = 0.9
    iteration_count = 0
    max_iteration_count = 10
    target_width
    target_height
    target_image
    current = start
    
    # Load the target file here and set target_image, target width and target height with the loaded image

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
            score = await compare(current_context, target_image, target_width, target_height)
            if (score > threshold):
                print("Successfully edited component to threshold")
                return
            current = await modify(current, iteration_count)
            iteration_count += 1
        print("Failed to reach goal by max iteration count")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agentic Frontend Editor')
    parser.add_argument('--target', type=str, default='./test_case/1/target.png',
                      help='Path to target output)')
    parser.add_argument('--start', type=str, default='./test_case/1/start.html',
                      help='Path to starting input)')
    
    args = parser.parse_args()
    print("Starting edit loop...")
    asyncio.run(main(args.target, args.current))


def async compare(current_context, current_url, target_image, target_width, target_height, current_iteration):
    page = await context.new_page()
    await page.goto(current_url)
    # Change this so that the page takes a screenshot with the passed dimensions instead
    screenshot = await page.screenshot(path=f"./qa/{current_iteration}.png", full_page=True)
    # Compare the target image to the current context

    score = calculate_image_hash_similarity(screenshot, target_image)
    return score

def calculate_image_hash_similarity(img1, img2):
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    hash_1 = imagehash.average_hash(img1)
    hash_2 = imagehash.average_hash(img2)
    return hash_1 - hash_2

def async segment():
    pass

def async modify(current, iteration_count):
    if (iteration_count == 0):
        # Init the loop by making a copy of the start html file with the suffix ITERATION_TOKEN + iteration_count and saving it.
        # Set current to this new url
    else:
        # create a new file by splitting the iteration token, remaking it with the new iteration count, and saving it
        
    
    # Once you have the new file, query cursor to modify it with the prompt "modify this html file such that it matches the target image more"

    # return the new file url







# Lets start with a target image of a component and an empty html file.
# Assume for now that the target image has the dimensions of the view port and a single button at the center
# The goal is for this program to start with an empty html file and correctly construct a button that is pixel perfect 
# relative to the target image.
# So the program will be driven through a loop of compare --> modify --> compare --> modify
# Compare should use image comparison to compare the target and the current html
# Keep track of the current similarity score of the button and the target, exit the program when it converges 
# under some set threshold

# ideally, this thing could recursively segment areas of difference, but that is a whole can of worms