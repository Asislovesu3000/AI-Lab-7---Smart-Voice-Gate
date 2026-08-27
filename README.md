# 🎙️ Smart Voice Gate

An AI-powered voice authentication system that uses speaker recognition to grant or deny access — like a smart gate controlled entirely by your voice.

---

## 📌 Features

- 🔊 **Voice Enrollment** — Register users by capturing voice samples
- 🧠 **Speaker Embeddings** — Uses the ECAPA-TDNN model (SpeechBrain) for voice fingerprinting
- 🔐 **Voice Authentication** — Verifies identity in real-time using cosine similarity
- 🗣️ **Speech Recognition** — Transcribes spoken commands using Whisper / SpeechRecognition
- 🖥️ **Streamlit UI** — Clean web-based interface for managing users and testing the gate

---

## 🗂️ Project Structure

```
Smart Voice Gate/
├── app.py                        # Main Streamlit application
├── voice_embedding.py            # Speaker embedding extraction
├── speech_recognition_module.py  # Speech-to-text module
├── user_manager.py               # User enrollment & management
├── gate_logic.py                 # Authentication decision logic
├── requirements.txt              # Python dependencies
├── pretrained_models/
│   └── spkrec-ecapa-voxceleb/
│       └── hyperparams.yaml      # Model config (weights excluded)
└── enrolled_users/
    └── users.json                # Enrolled user metadata (schema example)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Asislovesu3000/AI-Lab-7---Smart-Voice-Gate.git
cd AI-Lab-7---Smart-Voice-Gate
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the pretrained model
The ECAPA-TDNN speaker recognition model is required but not included in this repo due to its size (~83MB).

Download it automatically via SpeechBrain:
```python
from speechbrain.pretrained import SpeakerRecognition
SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                 savedir="pretrained_models/spkrec-ecapa-voxceleb")
```
Or run the app once — it will auto-download on first launch.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

---

## 🧪 How It Works

1. **Enroll** a user by recording their voice (3–5 samples recommended).
2. The system extracts a **voice embedding** (192-dim vector) using ECAPA-TDNN.
3. On authentication, the spoken voice is compared with stored embeddings via **cosine similarity**.
4. If similarity exceeds the threshold → **ACCESS GRANTED** ✅, else → **ACCESS DENIED** ❌.

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key libraries:

| Library | Purpose |
|---|---|
| `speechbrain` | Speaker embedding model |
| `streamlit` | Web UI |
| `torch` | Deep learning backend |
| `SpeechRecognition` | Voice-to-text |
| `pyaudio` | Microphone input |

---

## 📁 Notes

- `enrolled_users/*.pt` files (voice embeddings) are excluded from Git — they are generated at runtime when you enroll users.
- `pretrained_models/**/*.ckpt` files are excluded — download as described above.

---

## 👨‍💻 Author

**Asislovesu3000** — AI Lab Mini Project, Semester 6


# AI-Lab-7---Smart-Voice-Gate
The system is designed to control a physical gate using AI-based voice input and authentication ( Speech Recognition and Speaker Verification ).  A person approaches the gate and provides the required authentication, such as:  Voice and Word/password .  If both matches: Open the Gate , Else Show Alerts

# To Enroll Users
📝 Instructions <br>
Enter your name <br>
Type your custom passphrase (e.g., "open sesame") <br>
Click the mic and say your passphrase clearly <br>
Click Enroll to save your voiceprint <br>

# To Verify Voice and Pass Phrase 
🛡️ How It Works<br>
Select your enrolled name<br>
Record yourself saying your passphrase<br>
AI checks WHO you are (voice match)<br>
AI checks WHAT you said (phrase match)<br>
Both must pass to unlock! 🔓<br>
