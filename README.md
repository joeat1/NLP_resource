# NLP_resource
Some useful resource for NLP

> 跟踪自然语言处理（NLP）的进度: https://nlpprogress.com/
> [NLP 的巨人肩膀](https://zhuanlan.zhihu.com/p/50443871) ：较为详细的讲述了自然语言处理的发展历程

## Tools

> 主要包括 分词、词性标注等功能的 python 库

+ [spacy](https://spacy.io/)
+ [stanfordnlp](https://stanfordnlp.github.io/stanfordnlp/)     Python NLP Library for Many Human Languages
+ 结巴分词
+ [Interactive Attention Visualization](https://github.com/SIDN-IAP/attnvis)
+ [THULAC](https://github.com/thunlp/THULAC)    包括中文分词、词性标注功能。
+ [TextGrapher](https://github.com/liuhuanyong/TextGrapher)     输入一篇文档，形成对文章语义信息的图谱化展示。

## Pre-train

> 训练模型相关

+ [transformers](https://github.com/huggingface/transformers)  
    + 注意： `transformers > 3.1.0` 的版本下，在 `from_pretrained` 函数调用中添加 `mirror` 选项，如 `AutoModel.from_pretrained('bert-base-uncased', mirror='tuna')` 可以加快模型的下载。
    + 加上 `cache_dir="XXX"` 手动设置缓存地址，如果不设置，默认下载在 `~/.cache/torch` 或者 `C:\Users\XXXX\.cache\torch`，每个文件都有一个json作为标记，告知对应文件的作用。
+ [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
+ [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)     Pre-Training with Whole Word Masking for Chinese BERT

> 评价榜单

+ [GLUE](https://gluebenchmark.com/leaderboard)

> 训练技巧

+ [深度学习网络调参技巧](https://zhuanlan.zhihu.com/p/24720954)

## Dataset

+ [The Big Bad NLP Database](https://datasets.quantumstat.com/)
+ [CLUEDatasetSearch](https://github.com/CLUEbenchmark/CLUEDatasetSearch)

### Fakenews
> Fake News Detection on Social Media: A Data Mining Perspective

+ [fakenewschallenge](http://www.fakenewschallenge.org/)
+ [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus)
+ [Chinese_Rumor_Dataset](https://github.com/thunlp/Chinese_Rumor_Dataset)
+ [Weibo dataset and two Twitter datasets](https://github.com/chunyuanY/RumorDetection)
+ [虚假新闻检测数据集](https://blog.csdn.net/Totoro1745/article/details/84678858)

### Event
+ [Integrated Crisis Early Warning System (ICEWS) Dataverse](https://dataverse.harvard.edu/dataverse/icews)
+ [GDELT](https://www.gdeltproject.org/data.html#rawdatafiles)

## papers list

+ [NLP-journey](https://github.com/msgi/nlp-journey)
+ [EventExtractionPapers](https://github.com/BaptisteBlouin/EventExtractionPapers)
