import os  
import re 
import time  
import codecs  
import numpy as np  
from pyhanlp import *  #处理包
from collections import Counter  

# 初始化HanLP组件
CorpusLoader = SafeJClass('com.hankcs.hanlp.corpus.document.CorpusLoader')  # 语料加载器
NatureDictionaryMaker = SafeJClass('com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker')  # 词典生成器

# 定义文件路径
corpus_path = "./data/msr_training.utf8"  # 训练语料路径
model_path = "./data/msr_correction_model"  # 模型保存路径

# 加载并处理语料中的句子
sentences = CorpusLoader.convert2SentenceList(corpus_path)  # 将语料转换为句子列表
print(sentences[10])  # 打印第10个句子作为样例查看

# 为所有未标注的词设置默认词性标签'n'(名词)
for sent in sentences:
    for word in sent:
        if word.label is None:  # 如果词性标签为空
            word.setLabel("n")  # 设置为名词
print(sentences[10])  # 再次打印样例查看变化

# 训练模型
t0 = time.time()  
maker = NatureDictionaryMaker()  
maker.compute(sentences)  
maker.saveTxtTo(model_path)  
t1 = time.time()
print("2-gram训练结束，时间=%.2f秒" % (t1 - t0)) 

# 配置词典路径
HanLP.Config.CoreDictionaryPath = model_path + ".txt"  
HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  
print("HanLP.Config.CoreDictionaryPath=%s" % HanLP.Config.CoreDictionaryPath) print("HanLP.Config.BiGramDictionaryPath=%s" % HanLP.Config.BiGramDictionaryPath) 
# 加载HanLP核心词典类
CoreDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreDictionary') CoreBiGramTableDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary') 
def caculate_weight(pre_word, de_word, pro_word):
    """计算'的/地/得'在上下文中的权重"""
        if CoreDictionary.getTermFrequency(pre_word) != 0:  # 如果前驱词存在
        pre_weight = CoreBiGramTableDictionary.getBiFrequency(pre_word, de_word) / CoreDictionary.getTermFrequency(
            pre_word)
    else:
        pre_weight = 0  # 前驱词不存在则权重为0

    # 计算目标词与后继词的共现概率
    if CoreDictionary.getTermFrequency(de_word):  # 如果目标词存在
        pro_weight = CoreBiGramTableDictionary.getBiFrequency(de_word, pro_word) / CoreDictionary.getTermFrequency(
            de_word)
    else:
        pro_weight = 0  # 目标词不存在则权重为0

    # 计算平均权重作为最终得分
    cost = (pre_weight + pro_weight) / 2
    return cost


# 候选词列表('的','地','得')
candidate_word_li = [u'的', u'地', u'得']

# 读取测试文件并评估模型
with codecs.open('./data/的地得词组练习.txt', 'rb', 'utf-8', 'ignore') as infile:  # 以UTF-8编码打开文件
    for line in infile:
        line = line.strip()  # 去除首尾空白
        if line:  # 跳过空行
            # 分割出前驱词、当前词和后继词
            pre_word, cur_word, pro_word = re.split(u'【|】', line)

            # 计算每个候选词的权重
            candidate_weight_li = []
            for candidate_word in candidate_word_li:
                candidate_weight_li.append(caculate_weight(pre_word, candidate_word, pro_word))

            # 选择权重最高的候选词作为预测结果
            predict_word = candidate_word_li[np.argmax(candidate_weight_li)]

            # 输出预测结果
            if predict_word == cur_word:  # 预测正确
                print("正确：", predict_word, line)
            else:  # 预测错误
                print("错误：", predict_word, line)
