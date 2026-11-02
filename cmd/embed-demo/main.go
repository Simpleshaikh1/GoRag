package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/simpleshaikh/askmycode/internal/embed"
	"github.com/simpleshaikh/askmycode/internal/vector"
)

var docs = []string{
	"A token bucket rate limiter refills tokens at a fixed rate and rejects requests when empty.",
	"JWT tokens carry signed claims so the server can verify a user without a session store.",
	"A worker pool runs a fixed number of goroutines that pull jobs from a channel.",
	"Redis speaks RESP, a simple text protocol over TCP.",
	"Kubernetes Deployments keep a desired number of pod replicas running.",
	"The weather today is sunny with a light breeze.",
}

func main() {
	query := "how do I stop clients from sending too many requests?"
	if len(os.Args) > 1 {
		query = strings.Join(os.Args[1:], " ")
	}

	ctx := context.Background()
	vecs, err := embed.NewOllama().Embed(ctx, append([]string{query}, docs...))
	if err != nil {
		log.Fatal(err)
	}
	queryVec, docVecs := vecs[0], vecs[1:]
	fmt.Printf("query: %q\nvector size: %d numbers per text\n\n", query, len(queryVec))

	type scored struct {
		score float64
		text  string
	}
	results := make([]scored, len(docs))
	for i, v := range docVecs {
		results[i] = scored{vector.Cosine(queryVec, v), docs[i]}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })

	for _, r := range results {
		fmt.Printf("%.3f  %s\n", r.score, r.text)
	}
}
