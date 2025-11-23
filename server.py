import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string
from torch.nn.utils.rnn import pad_sequence

from model import Encoder
from tokenizer import MyBPETokenizer

# ==========================================
# 1. Configuration & Setup
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "E:\\TimePass\\GullibleTransformer\\runs\\gc2_best_acc.pth"
TOKENIZER_PATH = ".\\data\\#synthetic\\tokenizer_v0" 
MAX_LEN = 1024                      
VOCAB_SIZE = 500                  
EMBEDDING_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
NUM_CLASSES = 2
DROPOUT = 0.0                     
BIAS = False

app = Flask(__name__)

# ==========================================
# 2. Load Resources (Run once on startup)
# ==========================================
print("Loading Tokenizer...")
tokenizer = MyBPETokenizer()
try:
    tokenizer.load(TOKENIZER_PATH)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Ensure the tokenizer path is correct.")

print("Loading Model...")
model = Encoder(
    n_layers=NUM_LAYERS,
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    bias=BIAS,
    num_heads=NUM_HEADS,
    output_size=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)

try:
    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode (disables dropout, etc.)
    print("Model loaded successfully!")
except Exception as e:
    print(f"\nCRITICAL ERROR LOADING WEIGHTS: {e}")
    print("Tip: If keys don't match, check if you saved the Embedding layer separately.\n")

# ==========================================
# 3. Frontend Template (HTML/JS)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: none;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen">

    <div class="w-full max-w-2xl p-8 bg-white rounded-xl shadow-2xl">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold text-gray-800">🕵️‍♀️ Fake News Detector</h1>
            <p class="text-gray-500 mt-2">Paste a news headline to verify its authenticity.</p>
        </div>

        <div class="space-y-4">
            <label class="block text-sm font-medium text-gray-700">News Headline / Article</label>
            <textarea id="newsInput" rows="5" 
                class="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
                placeholder="Ex: Aliens have landed in Times Square..."></textarea>
            
            <button onclick="checkNews()" 
                class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition duration-200 flex justify-center items-center gap-2">
                <span>Analyze Text</span>
                <div id="loader" class="loader"></div>
            </button>
        </div>

        <!-- Result Section -->
        <div id="resultCard" class="hidden mt-8 p-6 rounded-lg border-l-4 transition-all duration-500">
            <h2 id="resultTitle" class="text-2xl font-bold mb-2"></h2>
            <p id="resultConf" class="text-gray-700"></p>
            
            <div class="mt-4 w-full bg-gray-200 rounded-full h-2.5">
                <div id="confBar" class="bg-blue-600 h-2.5 rounded-full" style="width: 0%"></div>
            </div>
        </div>
    </div>

    <script>
        async function checkNews() {
            const text = document.getElementById('newsInput').value;
            const btn = document.querySelector('button');
            const loader = document.getElementById('loader');
            const resultCard = document.getElementById('resultCard');

            if (!text.trim()) return alert("Please enter some text!");

            // UI Loading State
            loader.style.display = 'block';
            btn.disabled = true;
            resultCard.classList.add('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                
                // Display Results
                displayResult(data);

            } catch (error) {
                console.error(error);
                alert("Error connecting to the model.");
            } finally {
                loader.style.display = 'none';
                btn.disabled = false;
            }
        }

        function displayResult(data) {
            const card = document.getElementById('resultCard');
            const title = document.getElementById('resultTitle');
            const conf = document.getElementById('resultConf');
            const bar = document.getElementById('confBar');

            card.classList.remove('hidden');
            
            // 0 = True, 1 = Fake (Based on your training mapping)
            const isFake = data.label_index === 1;
            const confidence = (data.confidence * 100).toFixed(2);

            if (isFake) {
                card.className = "mt-8 p-6 rounded-lg border-l-4 bg-red-50 border-red-500 text-red-900";
                title.innerText = "🚨 Likely FAKE News";
                bar.className = "bg-red-500 h-2.5 rounded-full";
            } else {
                card.className = "mt-8 p-6 rounded-lg border-l-4 bg-green-50 border-green-500 text-green-900";
                title.innerText = "✅ Likely TRUE News";
                bar.className = "bg-green-500 h-2.5 rounded-full";
            }

            conf.innerText = `Confidence: ${confidence}%`;
            bar.style.width = `${confidence}%`;
        }
    </script>
</body>
</html>
"""

# ==========================================
# 4. Routes
# ==========================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Tokenize
    encoded_sent = tokenizer.encode(text)
    
    # 2. Convert to Tensor
    input_tensor = torch.tensor(encoded_sent)

    # 3. Truncate if necessary (Pre-padding check)
    if len(input_tensor) > MAX_LEN:
        input_tensor = input_tensor[:MAX_LEN]

    # 4. Pad (We need a batch dimension)
    # Since we only have 1 item, we just unsqueeze, but we need to pad to MAX_LEN 
    # if the model relies on positional embeddings of a specific size, 
    # OR if we simply want to be consistent with training.
    # Here we create a list of 1 tensor and pad it.
    padded_tensor = pad_sequence([input_tensor], batch_first=True, padding_value=0)
    
    # Ensure it's not longer than MAX_LEN (pad_sequence doesn't truncate)
    if padded_tensor.shape[1] > MAX_LEN:
         padded_tensor = padded_tensor[:, :MAX_LEN]

    # Send to Device
    X = padded_tensor.to(DEVICE)

    # 5. Inference
    with torch.no_grad():
        logits, _ = model(X, targets=None) 
        
        probs = F.softmax(logits, dim=1)
        
        conf, predicted_class = torch.max(probs, dim=1)

    return jsonify({
        'label_index': int(predicted_class.item()),
        'confidence': float(conf.item()),
        'logits': logits.tolist()
    })

if __name__ == '__main__':
    print("🚀 Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)