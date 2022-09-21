from flask import Flask, request, make_response, jsonify
from flask_cors import CORS

import numpy as np
import torch
from torch import nn

import gluonnlp as nlp
from torch.utils.data import Dataset
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils.utils import get_tokenizer


app =Flask(__name__)
CORS(app)

 
tokenizer = get_tokenizer()
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))



def predict(query):
    max_len = 128
    batch_size = 64
    result =[]
    for i in range(len(query)):
        data = [query[i], '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

        model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long()
            segment_ids = segment_ids.long()

            valid_length = valid_length
            label = label.long()

            out = model(token_ids, valid_length, segment_ids)

            test_eval = []
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("슬픔")
                elif np.argmax(logits) == 1:
                    test_eval.append("당황")
                elif np.argmax(logits) == 2:
                    test_eval.append("분노")
                elif np.argmax(logits) == 3:
                    test_eval.append("무기력")
                elif np.argmax(logits) == 4:
                    test_eval.append("중립")
                elif np.argmax(logits) == 5:
                    test_eval.append("기쁨")
                elif np.argmax(logits) == 6:
                    test_eval.append("상처")

            output = test_eval[0]
            result.append(output)
    print(result)

    return result

@app.route("/call", methods=['POST'])
def test():
    print(request.is_json)
   # print(request.get_json.get('input'))
    query = request.get_json().get('input').split('\n')
    print(query)
    return make_response(jsonify({"sentimental":predict(query)}),200)

 
 
if __name__ == "__main__":
    model = torch.load('/home/ubuntu/model.pth', map_location=torch.device('cpu'))
    app.run(host="0.0.0.0", port="5000")
