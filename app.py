import os
from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# English to Vietnamese model
en_to_vi_model_path = 'D:/model'
en_to_vi_tokenizer_path = 'D:/model'

# Vietnamese to English model
vi_to_en_model_path = 'D:/translate4/PhoMT_model'
vi_to_en_tokenizer_path = 'D:/translate4/PhoMT_model'

# Translation function
def translate(text, src_lang, tgt_lang):
    if src_lang == 'en' and tgt_lang == 'vi':
        tokenizer = MarianTokenizer.from_pretrained(en_to_vi_tokenizer_path)
        model = MarianMTModel.from_pretrained(en_to_vi_model_path)
    elif src_lang == 'vi' and tgt_lang == 'en':
        tokenizer = MarianTokenizer.from_pretrained(vi_to_en_tokenizer_path)
        model = MarianMTModel.from_pretrained(vi_to_en_model_path)
    else:
        return "Translation not supported"

    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    input_text = request.form['text']
    src_lang = request.form['src_lang']
    tgt_lang = request.form['tgt_lang']
    translated_text = translate(input_text, src_lang, tgt_lang)
    return translated_text

if __name__ == '__main__':
    app.run(debug=True)
