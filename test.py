import requests

key = "AFIRqunpIdhGSQwygmKAX1DI3xDg4mxh1lhUJ8BqSVEnNTv5qv6pJQQJ99BHACYeBjFXJ3w3AAAYACOG3eDw"
endpoint = "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"

headers = {
    "Ocp-Apim-Subscription-Key": key,
    "Content-Type": "application/ssml+xml",
    "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
}

ssml = """<speak version='1.0' xml:lang='en-US'>
  <voice xml:lang='en-US' xml:gender='Female' name='en-US-AriaNeural'>
    Hello, this is Azure Text to Speech working perfectly.
  </voice>
</speak>"""

response = requests.post(endpoint, headers=headers, data=ssml)

with open("output.mp3", "wb") as f:
    f.write(response.content)

print("Saved to output.mp3")