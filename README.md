# 🖼️ SmartCaptionAI 

## 🚀 Overview  
This **AI-powered Image Caption Generator** uses deep learning models to:  
✅ **Generate Alt Text** for uploaded images using the **BLIP model**.  
✅ **Enhance Captions** with GPT-4 for more engaging descriptions.  
✅ **Analyze Sentiment** of the generated captions.  

This project demonstrates the power of **Generative AI** by combining **Hugging Face Transformers (BLIP)**, **OpenAI GPT-4**, and **Streamlit** for an interactive user experience.  

---

## 📌 Features  
- **Image Captioning:** Automatically generate alt text for images.  
- **Caption Enhancement:** Improve AI-generated captions with **GPT-4**.  
- **Sentiment Analysis:** Analyze emotions in the generated captions.  
- **User-Friendly UI:** Built with **Streamlit** for a clean and interactive experience.  

---

## 🛠️ Tech Stack  
- **Python 3.11+**  
- **Streamlit** – Web interface  
- **Hugging Face Transformers (BLIP)** – Image captioning  
- **OpenAI GPT-4** – Caption enhancement  
- **Sentiment Analysis Pipeline** – Emotion detection  
- **Dotenv** – Environment variable management  

---

## 📦 Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator
```

### 2️⃣ Create a virtual environment and activate it  
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up API Keys  
Create a `.env` file in the project directory and add:  
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ Usage  

### **Run the Streamlit app**  
```bash
streamlit run app.py
```

### **Upload an Image & Get Captions!**  
1. Upload an image (**JPG, PNG**).  
2. AI generates **alt text** using BLIP.  
3. GPT-4 **enhances the caption**.  
4. Sentiment analysis detects **positive/neutral/negative** tone.  

---

## 🎯 Example Output  

### **Uploaded Image:** 🖼️ (Dog in a Park)  
```plaintext
✅ Generated Alt Text: "A brown dog sitting on green grass with a happy expression."
✨ Enhanced Caption: "A joyful brown dog sits in a lush green park, basking in the sun."
🔍 Sentiment Analysis: Positive (Confidence: 92%)
```

---

## 🏆 Future Enhancements  
🔹 **Multimodal AI:** Use vision-language models like **Gemini**.  
🔹 **More Fine-Tuning:** Improve GPT-4 prompts for richer captions.  
🔹 **Downloadable Captions:** Save generated captions in **TXT/CSV format**.  

---

## 📜 License  
This project is licensed under the **MIT License**.  

---

## 🤝 Contributing  
1. **Fork** the repo.  
2. **Create** a new branch:  
   ```bash
   git checkout -b feature-xyz
   ```  
3. **Commit** changes:  
   ```bash
   git commit -m "Added XYZ feature"
   ```  
4. **Push**:  
   ```bash
   git push origin feature-xyz
   ```  
5. Open a **Pull Request**.  

---

## 📧 Contact  
For questions, reach out at **srijalattala@example.com** or connect on **GitHub (Srija-Lattala)**.  
