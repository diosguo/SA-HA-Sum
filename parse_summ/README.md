# 基于句法解析的神经网络文本摘要模型

# 文件结构

| 文件名                     | 说明                                      |
| :------------------------- | ----------------------------------------- |
| config.json                | 模型配置文件，内有相关参数                |
| main.py                    | 运行模型主程序                            |
| make_parse.py              | 将下载的原始stories内容进行句法解析并保存 |
| parse_summarization_model/ | 模型package                               |
| - attention.py             | 定义Banh..Attention模型                   |
| - decoder.py               | 定义解码器                                |
| - encoder.py               | 定义编码器                                |
| - model.py                 | 定义Seq2Seq模型                           |
| - parse_parse.py           | 将句法解析出的结果字符串转为树结构        |
| - vocab                    | 由CNN/DM原始数据构建的词典                |
| - vocab.py                 | 定义字典类，提供word2id等功能             |
| - vocab_tag                | CoreNLP中句法解析的标签字典集合           |
| url_lists/                 | 用于区分数据集的train/test/valid          |