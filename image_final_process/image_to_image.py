import io
import os
import warnings
import json
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
os.environ['STABILITY_HOST'] = 'api.stability.ai/v2beta/stable-image/control/structure'
os.environ['STABILITY_KEY'] = 's'

def extract_scenarios(character_json):
    content = character_json["content"]
    
    scenario_descriptions = [value for key, value in content.items() if key.startswith("paragraph_")]
    
    return scenario_descriptions


file_path = '/Users/enmingzhang/Desktop/VScode-Workspace/Image_Generation/stableDefusion/image_final_process/final.json'

with open(file_path, 'r') as file:
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
    style = "japanese anime style"
    img_file = "/Users/enmingzhang/Desktop/VScode-Workspace/Image_Generation/stableDefusion/image_final_process/4013156552.png"
    img = Image.open(img_file)
    answers2 = stability_api.generate(
        prompt=style + "dragon enjoy sunbath on beach",
        # prompt = style + " girl enjoy sunbath on beach",
        # prompt = style + "girl on a boat fishing",
        steps=40,
        # seed = 2388508153,
        init_image=img,
        width=768,
        height=1344,
        start_schedule = 0.9, # 0 - 1 Set the strength of our prompt in relation to our initial image.
        cfg_scale=9, # 0 - 10 Influences how strongly your generation is guided to match your prompt.
        sampler=generation.SAMPLER_K_DPMPP_2M
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