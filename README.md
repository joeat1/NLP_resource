# NLP_resource
Some useful resource for NLP

> 跟踪自然语言处理（NLP）的进度: https://nlpprogress.com/

## Tools
+ [spacy](https://spacy.io/)
+ [stanfordnlp](https://stanfordnlp.github.io/stanfordnlp/)     Python NLP Library for Many Human Languages
+ 结巴分词
+ [Interactive Attention Visualization](https://github.com/SIDN-IAP/attnvis)
+ [THULAC](https://github.com/thunlp/THULAC)    包括中文分词、词性标注功能。
+ [TextGrapher](https://github.com/liuhuanyong/TextGrapher)     输入一篇文档，形成对文章语义信息的图谱化展示。

## Pre-train
+ [transformers](https://github.com/huggingface/transformers)  注意：transformers > 3.1.0 的版本下，在 from_pretrained 函数调用中添加 mirror 选项，如AutoModel.from_pretrained('bert-base-uncased', mirror='tuna')可以加快模型的下载。加上cache_dir="XXX"手动设置缓存地址。
+ [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
+ [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)     Pre-Training with Whole Word Masking for Chinese BERT

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

## papers

+ [NLP-journey](https://github.com/msgi/nlp-journey)
+ [EventExtractionPapers](https://github.com/BaptisteBlouin/EventExtractionPapers)
