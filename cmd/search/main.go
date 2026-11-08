package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/simpleshaikh/askmycode/internal/embed"
	"github.com/simpleshaikh/askmycode/internal/store"
)

// How many lines of each result to show.
const previewLines = 6

func main() {
	k := flag.Int("k", 5, "number of results")
	dbURL := flag.String("db", envOr("DATABASE_URL", store.DefaultURL), "postgres connection URL")
	flag.Parse()
	if flag.NArg() == 0 {
		log.Fatal(`usage: go run ./cmd/search "how do I verify a JWT?"`)
	}
	query := strings.Join(flag.Args(), " ")

	ctx := context.Background()
	db, err := store.Open(ctx, *dbURL)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	qvec, err := embed.NewOllama().EmbedQuery(ctx, query)
	if err != nil {
		log.Fatal(err)
	}
	results, err := db.Search(ctx, qvec, *k)
	if err != nil {
		log.Fatal(err)
	}

	for i, r := range results {
		fmt.Printf("%d. %.3f  %s:%d-%d  %s %s\n", i+1, r.Score, r.File, r.StartLine, r.EndLine, r.Kind, r.Name)
		lines := strings.Split(r.Content, "\n")
		if len(lines) > previewLines {
			lines = append(lines[:previewLines], "...")
		}
		for _, l := range lines {
			fmt.Println("   │ " + l)
		}
		fmt.Println()
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
