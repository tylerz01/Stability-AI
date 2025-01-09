import io
import os
import warnings
import json
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
os.environ['STABILITY_HOST'] = 'https://api.stability.ai/v2beta/stable-image/generate/sd3'
os.environ['STABILITY_KEY'] = 'sk-vC3SlsoDRZOSexgT2QzoBlPBy7Tz6IY4ufF1qd2Ku7vaQZaO'
import json

def extract_character_info(character_json):
    character_info = character_json['content']
    
    outfit = character_info['Outfit_description']
    outfit_details = ', '.join(outfit[key]['description'] for key in outfit)
    
    character_details = (
        f"Name: {character_info['name']}, Age: {character_info['Age']}, "
        f"Gender: {character_info['gender']}, Occupation: {character_info['Occupation']}, "
        f"Outfit: {outfit_details}"
    )
    
    return character_details
            

def extract_scenarios(character_json):
    content = character_json["content"]
    
    scenario_descriptions = [value for key, value in content.items() if key.startswith("paragraph ")]
    
    return scenario_descriptions

file_path = '/Users/enmingzhang/Desktop/VScode-Workspace/Image_Generation/stableDefusion/POP/POP/final_output.json'

with open(file_path, 'r') as file:
    json_data = json.load(file)

character_details_string = extract_character_info(json_data)
print(json_data)

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation.
    # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)
# Set up our initial generation parameters.
answers = stability_api.generate(
    prompt= character_details_string,
    # "steps": 40,
    # "width": 768,
    # "height": 1344,
    # "seed": 0,
    # "cfg_scale": 5,
    # "samples": 1,
    # "style_preset": "anime",
    seed=0, # If a seed is provided, the resulting generated image will be deterministic.
                    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                    # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
    steps=40, # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale= 5.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=768, # Generation width, defaults to 512 if not included.
    height=1344, # Generation height, defaults to 512 if not included.
    sampler=1, # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
)

# Set up our warning to print to the console if the adult content classifier is tripped.
# If adult content classifier is not tripped, display generated image.
filename = ""
seedNum = 0
for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            global img
            img = Image.open(io.BytesIO(artifact.binary))
            filename = str(artifact.seed)+ ".png"
            seedNum = artifact.seed
            img.save(str(artifact.seed)+ ".png") # Save our generated images its seed number as the filename.
            
            

base_path = '/Users/enmingzhang/Desktop/VScode-Workspace/Image_Generation/stableDefusion/image_final_process/'
file_path = os.path.join(base_path, filename)
json_path = '/Users/enmingzhang/Desktop/VScode-Workspace/Image_Generation/stableDefusion/POP/POP/final_output.json'
with open(json_path, 'r') as file:
    json_data = json.load(file)
    
scenario_des = extract_scenarios(json_data)
count = 0
for senario in scenario_des:
    print(senario)
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'],
        verbose=True,
        engine="stable-diffusion-xl-1024-v1-0", 
    )
    style = "anime style"
    img_file = "/Users/enmingzhang/Desktop/VScode-Workspace/Image_Generation/stableDefusion/image_final_process/" + str(seedNum)+".png"
    img = Image.open(img_file)
    answers2 = stability_api.generate(
        prompt=style + senario,
        # prompt = style + " girl enjoy sunbath on beach",
        # prompt = style + "girl on a boat fishing",
        steps=50,
        seed = seedNum,
        # seed = 2388508153,
        init_image=img,
        width=768,
        height=1344,
        start_schedule=1, # 0 - 1 Set the strength of our prompt in relation to our initial image.
        cfg_scale=9, # 0 - 10 Influences how strongly your generation is guided to match your prompt.
        sampler=generation.SAMPLER_K_EULER_ANCESTRAL
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated image.
    for resp in answers2:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img2
                img2 = Image.open(io.BytesIO(artifact.binary))
                img2.save(str(artifact.seed)+"#"+ str(count) + "-img2img.png")
                count = count + 1