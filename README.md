
# ChineseSpanQA

This is a span-based system for Question Answering (QA) in Chinese
. Potential uses include online automated customer services, etc.

## Configure the Python environment

We use [Anaconda3](https://www.anaconda.com/) as default. You can create a conda environment by ```environment.yml```.

P.S. ```torch==1.9.0+cu111``` might need to be installed from the [official website](https://pytorch.org/get-started/previous-versions/).

## Download the model weights

Create a sub-directory named ```ckpt/``` and put ```epoch=9-step=14210.ckpt``` inside it. The ```.ckpt``` file is available on [百度网盘](https://pan.baidu.com/s/1JWJYZ81ntjJeK21t81If8g?pwd=49w0)(提取码:49w0).

## Run the Flask application
First, launch the application
```
flask run -p 8000
```
Then, query the system via a POST request. For example, in Python, run the following lines:
```
>>> resp = requests.post("http://localhost:8000/predict", files={"context_path": open("context.txt", "r", encoding="utf-8"), "question_path": open("question.txt", "r", encoding="utf-8")})
>>> resp.json()
{'answer': '1995年，率先在业内提出"六保"服务--保真、保质、保价、保换、保修、保洗，用优良的品质、优质的服务营造"中国钻石专家"的品牌形象。', 'question': '金伯利的
六项保障服务是什么？'}
```
