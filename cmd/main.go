package main

import (
	"context"
	"fmt"
	"log"
	"os"

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
		log.Fatalf("new store: %v\n", err)
	}
	loadFile(store, ctx)
	question := "kubernetes是什么，请使用英语详尽回答，并且回答内容大于500字"
	docs := search(store, ctx, question)
	//for _,doc := range docs {
	//fmt.Println(doc)
	//}
	ask(docs, ctx, question)

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
		fmt.Println("Error opening file: ", err)
		return
	}
	p := documentloaders.NewText(f)

	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = 500   // size of the chunk is number of characters
	split.ChunkOverlap = 30 // overlap is the number of characters that the chunks overlap
	docs, err := p.LoadAndSplit(context.Background(), split)

	if err != nil {
		fmt.Println("Error loading document: ", err)
	}

	log.Println("Document loaded: ", len(docs))
	store.AddDocuments(ctx, docs)
}

func search(store vectorstores.VectorStore, ctx context.Context, query string) []schema.Document {
	docs, err := store.SimilaritySearch(ctx, query, 5)
	if err != nil {
		fmt.Println("Error search: ", err)
		return nil
	}
	return docs
}

func ask(docs []schema.Document, ctx context.Context, question string) {
	llm, err := ollama.New(ollama.WithModel("deepseek-r1:8b"))
	if err != nil {
		log.Fatal(err)
	}
	rag_template := `
	You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

	<context>
	{context}
 	{{.context}}
	</context>

		Answer the following question:

	{{.question}}
`
	prompts := prompts.NewPromptTemplate(rag_template, []string{"context", "question"})
	chain1 := chains.NewLLMChain(llm, prompts)
	m := make(map[string]any)
	m["context"] = docs
	m["question"] = question
	res, err := chains.Call(ctx, chain1, m)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res)
}
