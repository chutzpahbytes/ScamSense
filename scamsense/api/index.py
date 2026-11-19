import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, send_from_directory
import joblib

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'scamsense_model.joblib')
_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    if path and os.path.exists(os.path.join(template_dir, path)):
        return send_from_directory(template_dir, path)
    return send_from_directory(template_dir, 'index.html')

@app.route('/api/check', methods=['POST'])
def check():
    data = request.get_json(force=True)
    text = (data or {}).get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    from scamsense_core import explain
    p, verdict, reasons = explain(text, get_bundle())
    return jsonify({'score': round(p * 100), 'verdict': verdict, 'reasons': reasons})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
