package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/simpleshaikh/askmycode/internal/chunk"
	"github.com/simpleshaikh/askmycode/internal/embed"
	"github.com/simpleshaikh/askmycode/internal/store"
)

// Folders that hold tooling, dependencies or build output, not your code.
var skipDirs = map[string]bool{
	".git": true, ".idea": true, ".vscode": true, "vendor": true, "node_modules": true, "bin": true,
}

// Files bigger than this are usually generated.
const maxFileBytes = 200_000

// How many chunks go to Ollama per request.
const embedBatch = 32

func main() {
	root := flag.String("root", "..", "folder whose projects to index; each subfolder is one project")
	dbURL := flag.String("db", envOr("DATABASE_URL", store.DefaultURL), "postgres connection URL")
	dry := flag.Bool("dry", false, "only chunk and print, no embedding or database")
	skip := flag.String("skip", "", "comma-separated project folders to leave out, e.g. HugoProject")
	flag.Parse()
	for _, p := range strings.Split(*skip, ",") {
		if p = strings.TrimSpace(p); p != "" {
			skipDirs[p] = true
		}
	}

	ctx := context.Background()
	var db *store.Store
	if !*dry {
		var err error
		db, err = store.Open(ctx, *dbURL)
		if err != nil {
			log.Fatal(err)
		}
		defer db.Close()
	}
	embedder := embed.NewOllama()

	absRoot, err := filepath.Abs(*root)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("indexing %s\n\n", absRoot)

	began := time.Now()
	var files, chunks int
	err = filepath.WalkDir(absRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}

		info, err := d.Info()
		if err != nil || info.Size() > maxFileBytes {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(absRoot, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		cs := chunk.File(rel, src)
		if len(cs) == 0 {
			return nil
		}
		if *dry {
			files++
			chunks += len(cs)
			fmt.Printf("%4d chunks  %s\n", len(cs), rel)
			for _, c := range cs {
				fmt.Printf("             %4d-%-4d %-7s %s\n", c.StartLine, c.EndLine, c.Kind, c.Name)
			}
			return nil
		}

		vecs, err := embedAll(ctx, embedder, cs)
		if err != nil {
			return fmt.Errorf("%s: %w", rel, err)
		}
		if err := db.ReplaceFile(ctx, projectOf(rel), rel, cs, vecs); err != nil {
			return fmt.Errorf("%s: %w", rel, err)
		}

		files++
		chunks += len(cs)
		fmt.Printf("%4d chunks  %s\n", len(cs), rel)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	if *dry {
		fmt.Printf("\n%d files, %d chunks (dry run, nothing stored)\n", files, chunks)
		return
	}

	total, err := db.Count(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\n%d files, %d chunks in %s. %d chunks in the database.\n",
		files, chunks, time.Since(began).Round(time.Second), total)
}

// embedAll embeds a file's chunks in batches and returns the vectors in order.
func embedAll(ctx context.Context, e *embed.Ollama, cs []chunk.Chunk) ([][]float32, error) {
	var vecs [][]float32
	for start := 0; start < len(cs); start += embedBatch {
		end := min(start+embedBatch, len(cs))
		texts := make([]string, 0, end-start)
		for _, c := range cs[start:end] {
			texts = append(texts, c.EmbedText())
		}
		batch, err := e.EmbedDocuments(ctx, texts)
		if err != nil {
			return nil, err
		}
		vecs = append(vecs, batch...)
	}
	return vecs, nil
}

// projectOf returns the first folder of a relative path: "JWT-AUTH/a/b.go" -> "JWT-AUTH".
func projectOf(rel string) string {
	if i := strings.Index(rel, "/"); i >= 0 {
		return rel[:i]
	}
	return "."
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
