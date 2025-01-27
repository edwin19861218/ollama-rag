package main

import (
	"context"
	"log"
	"os"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/milvus"
)

func main() {
	ctx := context.Background()
	store, err := newStore(ctx)
	if err != nil {
		log.Fatalf("init store error: %v\n", err)
	}
	loadFile(store, ctx)
	question := "kubernetes是什么，请使用英语详尽回答，并且回答内容大于500字"
	docs := search(store, ctx, question)
	correctiveDocs := gradeDocuments(docs, ctx, question)
	ask(correctiveDocs, ctx, question)
}
func newStore(ctx context.Context) (vectorstores.VectorStore, error) {
	llm, err := ollama.New(ollama.WithModel("nomic-embed-text:latest"))
	if err != nil {
		log.Fatal(err)
	}
	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatal(err)
	}
	idx, err := entity.NewIndexAUTOINDEX(entity.L2)
	if err != nil {
		log.Fatal(err)
	}

	milvusConfig := client.Config{
		Address: "http://localhost:19530",
	}
	// Create a new milvus vector store.
	store, errNs := milvus.New(
		ctx,
		milvusConfig,
		milvus.WithDropOld(),
		milvus.WithCollectionName("langchaingo_docs"),
		milvus.WithIndex(idx),
		milvus.WithEmbedder(embedder),
	)
	return store, errNs
}

func loadFile(store vectorstores.VectorStore, ctx context.Context) {
	f, err := os.Open("./data/k8s.txt")
	if err != nil {
		log.Fatal("Error opening file: ", err)
		return
	}
	p := documentloaders.NewText(f)

	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = 500   // size of the chunk is number of characters
	split.ChunkOverlap = 30 // overlap is the number of characters that the chunks overlap
	docs, err := p.LoadAndSplit(context.Background(), split)

	if err != nil {
		log.Fatal("Error loading document: ", err)
	}

	log.Println("Document loaded: ", len(docs))
	_, err = store.AddDocuments(ctx, docs)
	if err != nil {
		log.Fatal("Error adding document: ", err)
	}
}

func search(store vectorstores.VectorStore, ctx context.Context, query string) []schema.Document {
	docs, err := store.SimilaritySearch(ctx, query, 5)
	if err != nil {
		log.Fatal("Error search: ", err)
		return nil
	}
	return docs
}

// see https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/#create-index
func gradeDocuments(docs []schema.Document, ctx context.Context, question string) []schema.Document {
	log.Printf("grading docs, input docs %d \n", len(docs))
	llm, err := ollama.New(ollama.WithModel("deepseek-r1:8b"))
	if err != nil {
		log.Fatal(err)
	}
	systemTemplate := `
You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
	Give me the result with json format which includes two keys 'think' and 'result'
`
	userTemplate := `
Retrieved document: \n\n {{.document}} \n\n User question:{{.question}}
`
	ut := prompts.NewHumanMessagePromptTemplate(userTemplate, []string{"document", "question"})
	st := prompts.NewSystemMessagePromptTemplate(systemTemplate, nil)
	pros := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{ut, st})
	var correctiveDocs []schema.Document
	for index, v := range docs {
		chain1 := chains.NewLLMChain(llm, pros)
		m := make(map[string]any)
		m["document"] = v.PageContent
		m["question"] = question
		res, err := chains.Call(ctx, chain1, m)
		if err != nil {
			log.Println("grade call err", err)
			continue
		}
		if text, ok := res["text"]; ok {
			if val, ok := text.(string); ok {
				if strings.Contains(val, "\"result\": \"yes\"") {
					correctiveDocs = append(correctiveDocs, v)
					continue
				}
			}
		}
		log.Printf("doc NO.%d incorrect \n %v \n\n", index, res)
	}
	log.Printf("graded docs, output docs %d\n", len(correctiveDocs))
	return correctiveDocs
}

func ask(docs []schema.Document, ctx context.Context, question string) {
	llm, err := ollama.New(ollama.WithModel("deepseek-r1:8b"))
	if err != nil {
		log.Fatal(err)
	}
	ragTemplate := `
	You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

	<context>
 	{{.context}}
	</context>

		Answer the following question:

	{{.question}}
`
	pros := prompts.NewPromptTemplate(ragTemplate, []string{"context", "question"})
	chain1 := chains.NewLLMChain(llm, pros)
	m := make(map[string]any)
	m["context"] = docs
	m["question"] = question
	res, err := chains.Call(ctx, chain1, m)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(res)
}
