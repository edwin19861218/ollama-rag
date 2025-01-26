# 基于langchaingo和milvus的RAG demo

背景：golang demo，为了不跑python

# 依赖

## milvus

安装参考https://milvus.io/docs/install_standalone-docker.md

代码访问：demo中hardcode写了http://localhost:19530

## ollama模型

安装参考https://ollama.com/download

按需调整ollama地址（如远程），请参考https://github.com/ollama/ollama/blob/main/envconfig/config.go#L22

```
# ollama list

deepseek-r1:8b                    28f8fd6cdc67    4.9 GB    About an hour ago  模型，建议显存8g+

nomic-embed-text:latest           0a109f422b47    274 MB    6 weeks ago 英文嵌入
```