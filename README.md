
# 🧠 AI Moderated Text Generator (Hugging Face + Python)

This simple script demonstrates how to build a **moderated AI text generator** using **Hugging Face Transformers**.  
It accepts a user prompt, applies input/output moderation, and generates creative responses guided by a system prompt.

---

## 🚀 Features

✅ Accepts **user input** from command line  
✅ Adds a **system prompt** to guide AI behavior  
✅ Uses **Hugging Face Transformers** for text generation  
✅ Performs **input moderation** (blocks disallowed words before calling AI)  
✅ Performs **output moderation** (filters unsafe words in the AI’s response)  
✅ Runs easily in **Google Colab or local Python**  

---

## 🧩 Requirements

Make sure you have:
- Python 3.9+
- `transformers`
- `torch`
- (optional) Google Colab GPU runtime for faster performance

Install dependencies:

```bash
pip install transformers torch
```

---

## 🔐 Setup (Optional Hugging Face Token)

If you want to use larger or private models, create a **free Hugging Face account**:

1. Go to [https://huggingface.co](https://huggingface.co)
2. Create an account and generate a token under [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. In Colab, set it up using:

```python
from huggingface_hub import login
login("your_hf_token_here")
```

Public models can be used without a token.

---

## 🧠 Script Code

```python
from transformers import pipeline

# Initialize model
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

# Simple moderation
BANNED = ["kill", "bomb", "hack", "terror", "suicide"]

def violates(text):
    return any(b in text.lower() for b in BANNED)

# Get user input
prompt = input("Enter your prompt: ")

# Input moderation
if violates(prompt):
    print("❌ Your input violated the moderation policy.")
else:
    system_prompt = "You are a creative and kind poet who writes safe, inspiring, and imaginative responses."
    full_prompt = f"{system_prompt}
User: {prompt}
Assistant:"

    result = generator(
        full_prompt,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )

    response = result[0]["generated_text"].split("Assistant:")[-1].strip()

    if violates(response):
        for w in BANNED:
            response = response.replace(w, "[REDACTED]")
        print("⚠️ Output was moderated:")
    else:
        print("✅ Safe Response:")

    print(response)
```

---

## 🧪 Example

```
Enter your prompt: write a poem about saving the planet from aliens

✅ Safe Response:
In the sky they came with fire and light,
But Earth’s hearts shone ever bright...
```

---

## ⚙️ Tips

- Use GPU runtime in Colab for speed.
- First run may take time to download the model (~13GB).
- You can switch to a smaller model like `microsoft/phi-2` or `distilgpt2` to speed up.

---

## 📜 License

MIT License – for learning and demonstration purposes only.
