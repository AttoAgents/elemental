## Example usage

```python
# Simple text message (backward compatible)
text_message = Message(role="user", content="Hello, world!")

# Adding an image to a message
message = Message(role="user", content="What's in this image?")
message.add_image("https://example.com/image.jpg")

# Creating a message with both text and image directly
message = Message(
    role="user",
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "image": {"url": "https://example.com/image.jpg"}}
    ]
)

# Using base64 encoded image
message = Message(role="user", content="Analyze this image")
message.add_image({"base64": "base64_encoded_string_here"})

# Checking if a message contains images
if message.is_multimodal():
    print("This message contains images")

# Getting only the text content
text = message.get_text_content()
```

# Example Images

https://commons.wikimedia.org/wiki/Commons:Featured_pictures/Places/Architecture/Cityscapes#/media/File:Umbrellas_at_Caudan_Waterfront_Mall.JPG 

https://commons.wikimedia.org/wiki/Commons:Featured_pictures/Objects/Vehicles/Rail_vehicles#/media/File:CN_8015,_5690_and_5517_Hinton_-_Jasper.jpg

https://commons.wikimedia.org/wiki/File:Sunflowers_helianthus_annuus.jpg#file