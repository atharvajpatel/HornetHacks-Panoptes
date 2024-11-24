import base64
from groq import Groq

APIKEY = "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"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your .png image
image_path = "army-bottle.jpg"

# Encode the image to a base64 string
base64_image = encode_image(image_path)

# Initialize the Groq client
client = Groq(api_key=APIKEY)

prompt = "If you detect a water bottle in this image, reply 'Yes'. If not, respond 'No'. Respond in one word."

# Create a chat completion request
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

# Print the model's response
print(chat_completion.choices[0].message.content)