from flask import Flask, jsonify, request
from infer import *
app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        context_file = request.files['context_path']
        question_file = request.files['question_path']
        print(context_file, question_file)
        context = context_file.read().decode("utf-8")
        question = question_file.read().decode("utf-8")
        print(context, question)
        ans = get_prediction(context, question)
        return jsonify({'question': question, 'answer': ans})

CKPT_PATH = 'ckpt/epoch=9-step=14210.ckpt'
PRETRAINED_PATH = 'hfl/chinese-bert-wwm-ext'
    
tokenizer_cls = BertTokenizer if 'bert' in PRETRAINED_PATH else AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
model = ModelQA.load_from_checkpoint(CKPT_PATH, pretrained_path=PRETRAINED_PATH)
model.eval()

def get_prediction(context, question):
    input_ids_a = tokenizer.encode(
        text=context,
        padding='max_length',
        max_length=480,
        truncation=True,
    )
    input_ids_b = tokenizer.encode(
        text=question,
        padding='max_length',
        max_length=32,
        truncation=True,
    )
    input_ids = input_ids_a + input_ids_b
    # assert len(input_ids) <= 512
    input_tensor = torch.LongTensor(input_ids).reshape(1, -1)
    st_scores, ed_scores = model(input_tensor=input_tensor)
    st_pred, ed_pred = torch.argmax(st_scores).item(), torch.argmax(ed_scores).item()
    # print(st_pred, ed_pred)
    if ed_pred > st_pred:
        answer = tokenizer.decode(token_ids=input_ids[st_pred:ed_pred+1])
        answer = answer.replace(' ', '')
    else:
        answer = NULL_PRINT
    return answer
        
if __name__ == '__main__':
    app.run()