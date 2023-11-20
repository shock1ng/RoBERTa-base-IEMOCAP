from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import re

# 指定本地模型路径 这两个是服务bert用的
model_path = "roBERTa-base"

# 加载tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained(model_path)
MAX_LEN = 128
# 读取CSV文件
data_frame = pd.read_csv("IEMOCAP_sentence_trans.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class getRoBert(nn.Module):
    def __init__(self):
        super(getRoBert, self).__init__()
        self.FC = nn.Linear(768,4)
        self.bert = RobertaModel.from_pretrained(model_path)

    def forward(self, spec_id, args):
        input_ids_list = []
        attention_mask_list = []
        for item in spec_id:
            # 查询给定编号对应的文本和标签
            result = data_frame[data_frame['id'] == item]
            if len(result) == 0:
                return None, None
            text = result['transcription'].values[0]
            # 使用正则表达式去除标点符号
            text = re.sub(r'[^\w\s]', '', text)

            # 步骤3: 对输入文本进行tokenization并添加特殊token
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN,
                                 return_tensors='pt')

            input_ids = encoding.get('input_ids').squeeze().to(device)  # 把[1,128]变成[128]
            attention_mask = encoding.get('attention_mask').squeeze().to(device)  # 把[1,128]变成[128]

            # 把两位append进列表中，等待下一步的堆叠
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        input_ids = torch.cat(input_ids_list, dim=0).view(args.batch_size, MAX_LEN)  # 堆叠成[batch,MAX_LEN]
        attention_mask = torch.cat(attention_mask_list, dim=0).view(args.batch_size, MAX_LEN)  # 堆叠成[batch,MAX_LEN]

        output = self.bert(input_ids, attention_mask)  # 改换思路，把矩阵丢进bert
        x = output.pooler_output  # 拿取pooler层的输出 ,large是[batch,1024]和base是[batch,768]

        if args.cuda:
            x = x.cuda()
        x = self.FC(x)
        return x


########以下是torch堆叠的完整代码，可以运行，但是acc不会变###########
'''
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel,RobertaForSequenceClassification, AdamW
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
import pandas as pd

# 指定本地模型路径 这两个是服务bert用的
model_path = "roBERTa-base"

# 加载tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# 读取CSV文件
data_frame = pd.read_csv("IEMOCAP_sentence_trans.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实现归一化操作的z_score
def z_score_scaling(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

# 实现归一化操作
def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# 文本最大长度512
MAX_LEN = 512

class getRoBert(nn.Module):  # 这里拿到（batchsize，100）形状的
    def __init__(self):
        super(getRoBert, self).__init__()
        self.FL = nn.Flatten()
        self.norm = nn.BatchNorm1d(768)
        self.FC1 = nn.Linear(768,100)   #这里如果是BertForSequenceClassification，那就左边是num_labels的值，如果不是（base=768,large=1024）
        self.FC2 = nn.Linear(100,4)
        self.FC3 = nn.Linear(4, 4)

        # self.bert = BertForSequenceClassification(BertConfig.from_pretrained(model_path , num_labels = 500))  # 尝试使用BertConfig加载权重，有所提升，但也只是25变35而已
        self.bert = RobertaModel.from_pretrained(model_path)
        # self.bert = BertForSequenceClassification.from_pretrained(model_path, num_labels=500)
    def forward(self, spec_id, args):
        output_list = []
        for item in spec_id:
            # 查询给定编号对应的文本和标签
            result = data_frame[data_frame['id'] == item]
            if len(result) == 0:
                return None, None
            text = result['transcription'].values[0]

            # 步骤3: 对输入文本进行tokenization并添加特殊token
            preChuLi = tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        max_length=MAX_LEN,  # 截断或者填充的最大长度512
                                        padding='max_length',
                                        # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
                                        truncation=True,  #达到最大长度会被截断
                                        return_attention_mask=True,  # 返回 attention mask
                                        return_tensors="pt"
                                        )

            input_ids = preChuLi.get('input_ids')
            mask_att = preChuLi.get('attention_mask')  # 这个用来标记单词是真实的还是填充得到的单词，真实是1，填充的是0

            input_ids = input_ids.cuda()  # 自从在上面把model和这个放在cuda上之后，训练速度飞速提升！！！显存占用多了600兆
            mask_att = mask_att.cuda()

            output = self.bert(input_ids, mask_att)  #出来是[1,4],因为bert被定义在gpu上，所以后面的生成也都在gpu上
            output = output[1]   # 不用BertForSequenceClassification时采用,这里出来[1,768]
            # output = output.logits  #获取主要的数据   用BertForSequenceClassification时采用
            # lst_out = output.tolist()  #转换成list数据
            # tmp_out = z_score_scaling(lst_out)  # 进去归一化 二选一，这俩效果差不多

            # 计算均值和标准差，手动实现z-scores归一化
            mean = torch.mean(output, dim=1)
            std = torch.std(output, dim=1)
            normalized_data = (output - mean.unsqueeze(1)) / std.unsqueeze(1)

            output_list.append(normalized_data)

        # output = np.stack(output_list, axis=0)  # batch个[1，4]堆叠变成[128，1，4]
        output = torch.cat(output_list, dim=0).view(args.batch_size, 1,768)  #128是batch，1*768是出bert的形状,堆叠成[batch,1,768]
        # output = torch.tensor(output)  # 把列表变成张量数据类型
        # print(output.shape)                 # torch.Size([batch, 1, 768])

        # 直接输出
        if args.cuda:
            output = output.cuda()
        output = self.FL(output)
        # output = self.norm(output)
        output = output.to(self.FC1.weight.dtype)
        output = self.FC1(output)
        output = self.FC2(output) # 出torch.Size([128, 4])
        output = self.FC3(output)
        return output
'''