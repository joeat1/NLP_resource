# :tada: NLP_resource :confetti_ball:
Some useful resource for NLP

> 跟踪自然语言处理（NLP）的进度: https://nlpprogress.com/
> [NLP 的巨人肩膀](https://zhuanlan.zhihu.com/p/50443871) ：较为详细的讲述了自然语言处理的发展历程

## 基本工具

> 主要包括分词、词性标注、命名实体识别等功能的 python 库

+ [spacy](https://spacy.io/)
+ [stanfordnlp](https://stanfordnlp.github.io/stanfordnlp/) ：Python NLP Library for Many Human Languages
+ [Jieba 结巴分词](https://github.com/fxsjy/jieba)：强大的Python 中文分词库
+ 
+ [Interactive Attention Visualization](https://github.com/SIDN-IAP/attnvis)
+ [THULAC](https://github.com/thunlp/THULAC)    包括中文分词、词性标注功能。
+ [TextGrapher](https://github.com/liuhuanyong/TextGrapher)     输入一篇文档，形成对文章语义信息的图谱化展示。

## 文本表示

+ [句嵌入](https://github.com/Separius/awesome-sentence-embedding)

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

## 数据集

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

## 论文列表
> 顶会：[ACL](https://2021.aclweb.org/)、[EMNLP](https://2021.emnlp.org/)、NAACL


+ [NLP-journey](https://github.com/msgi/nlp-journey)
+ [EventExtractionPapers](https://github.com/BaptisteBlouin/EventExtractionPapers)

## 主要研究机构
> 如有信息不正确或缺失，欢迎批评指正并留言，列表将定期更新。
> 
> PS：此处排名不分先后，排名请看 [CSRankings](http://csrankings.org/#/index?nlp&world)

![[国内NLP传承图 知乎用户提供](https://www.zhihu.com/question/24366306)](res/img/NLP_researchers.jpg "国内NLP传承图 知乎用户提供")

|  名称  |  GitHub  | 备注 |
|--|--|--|
| 高校 |
| [斯坦福大学自然语言处理研究组 Stanford NLP](http://nlp.stanford.edu/) | https://github.com/stanfordnlp| Stanford CoreNLP |
| [卡耐基梅隆大学语言技术中心](https://www.lti.cs.cmu.edu/) | | |
| [北京大学计算语言学研究所](https://icl.pku.edu.cn/) | 语言计算与机器学习组 https://github.com/lancopku | 计算语言学教育部重点实验室 |
| [清华大学自然语言处理与社会人文计算实验室](http://nlp.csai.tsinghua.edu.cn/) | https://github.com/thunlp | 孙茂松、刘知远团队
| [哈工大社会计算与信息检索研究中心 SCIR](http://ir.hit.edu.cn/) | https://hub.fastgit.org/HIT-SCIR |刘挺团队|
| [中科院计算所自然语言处理研究组](http://nlp.ict.ac.cn/ictnlp_website/) | https://github.com/ictnlp | |
| [中国科学院软件研究所中文信息处理实验室](http://www.icip.org.cn/zh/homepage/) | | |
| [复旦大学自然语言处理实验室](https://nlp.fudan.edu.cn/) |  https://github.com/FudanNLP |
| [南京大学自然语言处理研究组](http://nlp.nju.edu.cn/homepage/) | | 微信号 NJU-NLP |
| [香港科技大学人类语言技术中心](http://www.cse.ust.hk/~hltc/) | | |
| [爱丁堡大学自然语言处理小组(EdinburghNLP)](https://edinburghnlp.inf.ed.ac.uk/) | https://github.com/EdinburghNLP/ | |
| 企业 |
| [腾讯人工智能实验室](https://ai.tencent.com/ailab/nlp/zh/index.html) | |
| [微软亚研自然语言计算组](https://www.microsoft.com/en-us/research/group/natural-language-computing/) | | |
| [百度自然语言处理](https://nlp.baidu.com/homepage/index) | https://github.com/baidu | 提供 [PaddlePaddle](https://github.com/PaddlePaddle) 架构 |
| [搜狗实验室](http://www.sogou.com/labs/) | | 提供[预料资源](http://www.sogou.com/labs/resource/list_pingce.php) |
| [阿里巴巴达摩院语言技术实验室](https://damo.alibaba.com/labs/language-technology) | | |

## 信息资讯
+ 机器学习算法与自然语言处理 微信公众号和[知乎专栏](https://www.zhihu.com/column/qinlibo-ml)


## 基础知识/训练
+ [2021年47个机器学习项目](https://data-flair.training/blogs/machine-learning-project-ideas/)
+ [Paddle](https://github.com/PaddlePaddle/Paddle)