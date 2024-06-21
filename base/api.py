#!/usr/bin/env python3
#
import sys, os, pdb
import random, time
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 # Use the API
# TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.getenv("OPENAI_ORG_ID"))'
openai.organization = os.getenv("OPENAI_ORG_ID")

import google.generativeai as genai
genai.configure(api_key=os.environ["API_KEY"])

def openai_api_chat(args, model=None, input_seq=None, system_prompt=None, temperature=None):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_seq},
        ]
    else:
        messages = [
            {"role": "user", "content": input_seq},
        ]
    model = model if model is not None else args.model_name_or_path
    if temperature is None: temperature = args.temperature

    for delay_secs in (2**x for x in range(10)):
        try:
            response = client.chat.completions.create(model=model,  # assuming the GPT-4 model identifier
            temperature=temperature,
            max_tokens=args.max_length,
            top_p=args.top_p,
            n=1,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            messages=messages)
            break

        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue

    output_seq = response.choices[0].message.content
    return output_seq


def gemini_api_complete(model=None, input_seq=None, system_prompt=None):
    # model = genai.GenerativeModel(
    #     model_name="gemini-1.5-pro-latest",
    #     system_instruction=system_prompt
    # )
    model = genai.GenerativeModel("gemini-pro")
    response = ""
    for delay_secs in (2**x for x in range(10)):
        try:
            # response = model.generate_content(input_seq)
            response = model.generate_content(system_prompt + "\n\n" + input_seq)
        except Exception as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
    if response:
        return response._result.candidates[0].content.parts[0].text
    else:
        return ""