import openai
import json
import time
from tqdm import tqdm
from openai import OpenAI

def generate_user_prompts(num_prompts=200, output_path='generated_prompts.json', save_interval=30):
    # Initialize OpenAI client
    client = OpenAI(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1"
    )

    # Load existing prompts from JSON file if it exists
    try:
        with open(output_path, 'r') as f:
            existing_prompts = json.load(f)
    except FileNotFoundError:
        existing_prompts = []

    existing_prompt_texts = set(item['user_input'] for item in existing_prompts)

    system_prompt = """You are an AI assistant tasked with generating prompts for a molecule generation model. The model should be able to generate molecules with desired properties given from the user. The user prompt examples you are generating should vary in their length and complexity. Each prompt should be a single sentence, ending with a period."""

    example_prompts = [
        "Generate a molecule that can inhibit the replication of a specific RNA virus without affecting host cell function.",
        "Design a protein that can act as an efficient catalyst for the degradation of environmental pollutants like plastics.",
        "Create a molecule that selectively targets cancer cells by binding to overexpressed surface receptors, minimizing side effects on healthy cells.",
        "Develop a protein that can mimic the effects of insulin and regulate blood sugar levels in diabetic patients.",
        "Generate a small molecule that can cross the blood-brain barrier and alleviate neuroinflammation in Alzheimer's disease."
    ]

    new_prompts_added = 0

    for i in tqdm(range(num_prompts), desc="Generating prompts"):
        # Prepare the messages with the system prompt and existing examples
        messages = [{"role": "system", "content": system_prompt}]

        # Append previously generated prompts to the messages for context
        for prompt in existing_prompts[-10:]:  # Limit to the last 10 prompts to keep context manageable
            messages.append({"role": "assistant", "content": prompt['user_input']})
            messages.append({"role": "user", "content": "Thank you. Please provide another unique example."})

        # Add fixed example prompts for initial diversity
        for example in example_prompts[:3]:
            messages.append({"role": "assistant", "content": example})
            messages.append({"role": "user", "content": "Thank you. Please provide another unique example."})

        try:
            response = client.chat.completions.create(
                model="lmstudio-community/mathstral-7B-v0.1-GGUF/mathstral-7B-v0.1-Q5_K_M.gguf",
                messages=messages,
                temperature=0.9,
                max_tokens=200
            )
            
            new_prompt = response.choices[0].message.content.strip()
            
            # Check for uniqueness and validity of the new prompt
            if new_prompt not in existing_prompt_texts and new_prompt.endswith('.'):
                existing_prompts.append({"user_input": new_prompt, "assistant": ""})
                existing_prompt_texts.add(new_prompt)
                new_prompts_added += 1
                
                # Save prompts to JSON file after every 'save_interval' successful generations
                if (i + 1) % save_interval == 0:
                    with open(output_path, 'w') as f:
                        json.dump(existing_prompts, f, indent=2)
                    print(f"Saved {new_prompts_added} prompts after {i + 1} iterations.")
            
            time.sleep(1)  # Add a small delay to avoid overwhelming the API
        except Exception as e:
            print(f"Error generating prompt: {e}")
            time.sleep(5)  # Longer delay if there's an error

    # Final save
    with open(output_path, 'w') as f:
        json.dump(existing_prompts, f, indent=2)

    print(f"{new_prompts_added} new prompts generated and saved to {output_path}")

# Set the number of prompts to generate and output path
num_prompts = 200
output_path = 'generated_prompts.json'
save_interval = 30

# Run the function to generate prompts
generate_user_prompts(num_prompts, output_path, save_interval)