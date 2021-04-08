# :tada: NLP_resource :confetti_ball:
> [NLP 的巨人肩膀](https://zhuanlan.zhihu.com/p/50443871) ：较为详细的讲述了自然语言处理部分研究的发展历程 

### :balloon: Contents

| | |
| -- | -- |
| 论文列表 [:arrow_heading_down:](#论文列表) | 研究内容 [:arrow_heading_down:](#研究内容) |
| 工具库 [:arrow_heading_down:](#工具库) |  |

## :mortar_board: 论文列表

> 顶会：[ACL](https://aclweb.org/)、[EMNLP](https://2021.emnlp.org/)、[NAACL](https://naacl.org/)、COLING （前三个会议的录用数是 CSRankings 在本领域的评价指标）
>
> 部分会议的[历年录用率](https://aclweb.org/aclwiki/Conference_acceptance_rates) 


+ [NLP-journey](https://github.com/msgi/nlp-journey) 
+ [EventExtractionPapers](https://github.com/BaptisteBlouin/EventExtractionPapers) 

## :boom:  研究内容

> - [自然语言处理-概述](https://nlpoverview.com/) 应用于自然语言深度学习的技术概述，包括理论，实现，应用和最先进的结果。 

### 文本表示

> 自然语言处理中针对输入文本的第一步操作就是将 word/sentence/document 用**低维紧致向量**来表示，即嵌入（embedding）
>
> 以下只列出常用的模型算法，其他大量的论文和代码请从 [嵌入相关论文和代码](https://github.com/Separius/awesome-sentence-embedding) 中查阅。

+ 浅层嵌入
  + [word2vec](https://arxiv.org/abs/1301.3781)
  + [Glove](https://nlp.stanford.edu/pubs/glove.pdf)
  + [sense2vec](https://arxiv.org/abs/1511.06388) 
  + [paragraph_vector](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)  文档和句子的分布式表达

+ 上下文嵌入
  + [ElMo](https://arxiv.org/abs/1802.05365) 
  + [ULMFiT](https://arxiv.org/abs/1801.06146)  通用语言模型进行文本分类微调
  + [BERT](https://arxiv.org/abs/1810.04805) 
  + [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 



[返回目录 :arrow_heading_up:](#Contents) 

##  :cupid: 工具库

> 分词、词性标注、命名实体识别等功能的工具，主要为 Python、Java 语言

+  [NLTK](http://www.nltk.org/) - 自然语言工具包 :+1:

+ [spacy](https://spacy.io/) - 使用 Python 和 Cython 的高性能的自然语言处理库  :+1:

+ [gensim](https://radimrehurek.com/gensim/index.html) - 用于对纯文本进行无监督的语义建模的库，支持 word2vec 等算法 :+1:

+ [StanfordNLP](https://nlp.stanford.edu/software/index.shtml)  - 适用多语言的 NLP Library ，包含 Java 和 Python 语言 :+1:

+ [OpenNLP](https://opennlp.apache.org/) - 基于机器学习的自然语言处理的工具包，使用 Java 语言开发 :+1:

+ [TextBlob](http://textblob.readthedocs.org/) - 为专研常见的自然语言处理（NLP）任务提供一致的 API 

  

+ [Jieba 结巴分词](https://github.com/fxsjy/jieba) - 强大的Python 中文分词库 :+1:

+ [SnowNLP](https://github.com/isnowfy/snownlp) - 中文自然语言处理 Python 包，没有用NLTK，所有的算法都是自己实现的

+ [FudanNLP](https://github.com/FudanNLP/fnlp) - 用于中文文本处理的 Java 函式库

+ [THULAC](https://github.com/thunlp/THULAC) - 包括中文分词、词性标注功能。

> 预训练模型相关

+ [transformers](https://github.com/huggingface/transformers)  - 强大的预训练模型加载训练库:+1: 
  + 注意： `transformers > 3.1.0` 的版本下，在 `from_pretrained` 函数调用中添加 `mirror` 选项，如 `AutoModel.from_pretrained('bert-base-uncased', mirror='tuna')` 可以加快模型的下载。
  + 加上 `cache_dir="XXX"` 手动设置缓存地址，如果不设置，默认下载在 `~/.cache/torch` 或者 `C:\Users\XXXX\.cache\torch`，每个文件都有一个json作为标记，告知对应文件的作用。
+ [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 
+ [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)     Pre-Training with Whole Word Masking for Chinese BERT

> 深度学习架构

+ Pytorch
+ Tensorflow
+ [Paddle](https://github.com/PaddlePaddle/Paddle) 

> 其他

+ [Interactive Attention Visualization](https://github.com/SIDN-IAP/attnvis) - 交互式的注意力可视化

+ [TextGrapher](https://github.com/liuhuanyong/TextGrapher) - 输入一篇文档，形成对文章语义信息的图谱化展示。

[返回目录 :arrow_heading_up:](#Contents) 



## :cyclone: 数据集

+ [nlp-datasets](https://github.com/niderhoff/nlp-datasets) - 很好的自然语言资料集集合
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

[返回目录 :arrow_heading_up:](#Contents) 



## :fire: 主要研究机构
> 如有信息不正确或缺失，欢迎批评指正并留言，列表将定期更新。
>
> PS：此处排名不分先后，排名请看 [CSRankings](http://csrankings.org/#/index?nlp&world)。
>
> 下图为[国内NLP传承图](https://www.zhihu.com/question/24366306)：

![](res/img/NLP_researchers.jpg "国内NLP传承图 知乎用户提供")

<div align='center' > 国内NLP传承图 知乎用户提供 </div>



|  名称  |  GitHub  | 备注 |
|--|--|--|
| 高校 |
| [斯坦福大学自然语言处理研究组 Stanford NLP](http://nlp.stanford.edu/) | https://github.com/stanfordnlp| Stanford CoreNLP |
| [卡耐基梅隆大学语言技术中心](https://www.lti.cs.cmu.edu/) | | |
| [北京大学计算语言学研究所](https://icl.pku.edu.cn/) | 语言计算与机器学习组 https://github.com/lancopku | 计算语言学教育部重点实验室 |
| [清华大学自然语言处理与社会人文计算实验室](http://nlp.csai.tsinghua.edu.cn/) | https://github.com/thunlp | 孙茂松、刘知远团队|
| [哈工大社会计算与信息检索研究中心 SCIR](http://ir.hit.edu.cn/) | https://hub.fastgit.org/HIT-SCIR | 刘挺团队 |
| [中科院计算所自然语言处理研究组](http://nlp.ict.ac.cn/ictnlp_website/) | https://github.com/ictnlp |      |
| [中国科学院软件研究所中文信息处理实验室](http://www.icip.org.cn/zh/homepage/) |      |      |
| [复旦大学自然语言处理实验室](https://nlp.fudan.edu.cn/) | https://github.com/FudanNLP |
| [南京大学自然语言处理研究组](http://nlp.nju.edu.cn/homepage/) |      | 微信号 NJU-NLP |
| [香港科技大学人类语言技术中心](http://www.cse.ust.hk/~hltc/) |      |     |
| [爱丁堡大学自然语言处理小组(EdinburghNLP)](https://edinburghnlp.inf.ed.ac.uk/) | https://github.com/EdinburghNLP/ | |
| 企业 |
| [腾讯人工智能实验室](https://ai.tencent.com/ailab/nlp/zh/index.html) |   | | |
| [微软亚研自然语言计算组](https://www.microsoft.com/en-us/research/group/natural-language-computing/) |      |      |
| [百度自然语言处理](https://nlp.baidu.com/homepage/index) | https://github.com/baidu | 提供 [PaddlePaddle](https://github.com/PaddlePaddle) 架构 |
| [搜狗实验室](http://www.sogou.com/labs/) |      | 提供[预料资源](http://www.sogou.com/labs/resource/list_pingce.php) |
| [阿里巴巴达摩院语言技术实验室](https://damo.alibaba.com/labs/language-technology) |      |      |

[返回目录 :arrow_heading_up:](#Contents) 

## :loudspeaker: 信息资讯
+ 机器学习算法与自然语言处理 微信公众号和[知乎专栏](https://www.zhihu.com/column/qinlibo-ml) 
+ 跟踪自然语言处理（NLP）的进度: https://nlpprogress.com/

[返回目录 :arrow_heading_up:](#Contents) 

## :notebook: 基础知识/训练

> 课程学习/资料

+ 斯坦福大学-自然语言处理与深度学习-[CS224n](http://web.stanford.edu/class/cs224n/) :+1:
+ 斯坦福大学-自然语言理解-[CS224U](https://web.stanford.edu/class/cs224u) 
+ 斯坦福大学-机器学习-[CS229旧](https://see.stanford.edu/Course/CS229/)  [CS229-新](http://cs229.stanford.edu/) 
+ 马萨诸塞大学-高级自然语言处理-[CS 685](https://people.cs.umass.edu/~miyyer/cs685) 
+ 约翰霍普金斯大学-机器翻译-[EN 601.468/668](http://mt-class.org/jhu/syllabus.html) 
+ 麻省理工学院-深度学习-[6.S094， 6.S091， 6.S093](https://deeplearning.mit.edu/) 
+ 巴塞罗那 UPC-[语音和语言的深度学习](https://telecombcn-dl.github.io/2017-dlsl/) 
+ 麻省理工学院-线性代数-[18.06 SC](http://ocw.mit.edu/18-06SCF11) 
+ [用 Python 进行自然语言处理](https://www.nltk.org/book/) 
+ [2021年47个机器学习项目](https://data-flair.training/blogs/machine-learning-project-ideas/) 

> 博客

+ [详解 transformer]( https://jalammar.github.io/illustrated-transformer/) 
+ [ruder的博客](https://ruder.io/ ) 

> 评价榜单

+ [GLUE](https://gluebenchmark.com/leaderboard) 

> 训练技巧

+ [深度学习网络调参技巧](https://zhuanlan.zhihu.com/p/24720954) 


[返回目录 :arrow_heading_up:](#Contents) 

##  :white_large_square: ​ TODO

:white_check_mark: 

:white_large_square: 

## :page_with_curl: 参考

+ [emoji-list](https://github.com/caiyongji/emoji-list)
+ https://github.com/keon/awesome-nlp 
+ https://github.com/kmario23/deep-learning-drizzle 深度学习、强化学习、机器学习、计算机视觉和 NLP 相关讲座



## :pray: 贡献 

如果您找到适合本项目的任何类别的资料，则请提出问题或发送 PR 。

感谢为此项目提供帮助的成员和参考的资料。:gift_heart:

