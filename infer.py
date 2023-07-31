from model import *
import argparse
from transformers import BertTokenizer, AutoTokenizer
import torch

NULL_PRINT = '抱歉，关于这个问题我暂时没有可以提供的信息^_^'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='/root/ChineseSpanQA/ckpt/epoch=9-step=14210.ckpt')
    parser.add_argument('--pretrained_path', type=str, default='/root/chinese-bert-wwm-ext')
    args = parser.parse_args()
    
    tokenizer_cls = BertTokenizer if 'bert' in args.pretrained_path else AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    model = ModelQA.load_from_checkpoint(args.ckpt_path, pretrained_path=args.pretrained_path)
    model.eval()
    context = input('背景：')
    # context = '上海金伯利钻石集团有限公司 [1] 坚持"严谨务实、开拓创新"的企业精神，致力于每一个环节的精准把握。1995年，率先在业内提出"六保"服务--保真、保质、保价、保换、保修、保洗，用优良的品质、优质的服务营造"中国钻石专家"的品牌形象。其中公司独有的全国范围内终身"保换"服务更是想消费者所想，体现了金伯利"亲情化、人性化"的服务理念。 在金伯利的钻石世界里，璀璨是唯一的颜色，精雕细琢是最高的切工，真诚是最纯的净度，品质永远是沉甸甸的分量。作为专业的钻石首饰生产销售企业,公司的钻石切磨中心和首饰镶嵌中心，配有从国外引进的专业设备，各类专业技师、技术工人1000余人。从钻石切磨、首饰镶嵌到配货销售,整个生产流程严格按照国际标准操作，并由专职检验人员监督，经过层层检验，以确保产品质量。 金伯利始终相信专业人做专业事，坚持“质量是生产出来的”，严格按照企标及国标的规定，完善工艺流程，从生产的第一个环节加强质量管理。 生产过程中的每道工序，都由具有十年以上工作经验的技师严格把关，产品经过多次分检，最后总检，杜绝有任何问题的产品进入市场。 金伯利钻石均采用世界顶级的比利时完美切割工艺，每一颗都经过精确计算、精心设计和完美切割抛磨，所以钻石颗颗光芒闪耀、火彩照人。除了来自极品钻石产地南非之外，优质切工是决定金伯利钻石璀璨火彩的重要因素。金伯利钻石分布在全国各地的600多家专营店所经营的优质钻石，均采用世界顶级的比利时完美切割工艺，每一颗都经过精确计算、精心设计和完美切割抛磨，使光线充分折射出钻石顶面，所以钻石颗颗光芒闪耀、火彩照人。一颗钻石的价值不菲，除了它的有形价格，还包含着它的无形价值。'
    while True:
        if len(context) == 0:
            break
        question = input('问题：')
        # question = '金伯利公司的六项保障服务是什么？'
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
            print(answer)
        else:
            print(NULL_PRINT)