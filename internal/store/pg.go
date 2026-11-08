// Package store keeps chunks and their embeddings in Postgres with pgvector.
package store

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"

	"github.com/simpleshaikh/askmycode/internal/chunk"
)

const DefaultURL = "postgres://askmycode:askmycode@localhost:5433/askmycode"

// Dims must match the embedding model: nomic-embed-text returns 768 numbers.
// Switching models means changing this and re-indexing everything.
const Dims = 768

// Each row keeps the chunk text (what Claude reads) next to its embedding
// (what search compares). The HNSW index makes nearest-neighbour search fast
// without comparing the query against every row.
var schema = fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS chunks (
	id         bigserial PRIMARY KEY,
	project    text NOT NULL,
	file       text NOT NULL,
	kind       text NOT NULL,
	name       text NOT NULL,
	start_line int  NOT NULL,
	end_line   int  NOT NULL,
	content    text NOT NULL,
	embedding  vector(%d) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (file);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
`, Dims)

type Store struct {
	pool *pgxpool.Pool
}

// Open connects, enables the vector extension and creates the schema.
func Open(ctx context.Context, url string) (*Store, error) {
	// The extension has to exist before pgx can learn the vector type, so
	// create it on a plain connection first.
	conn, err := pgx.Connect(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("connecting to postgres (is `docker compose up -d` running?): %w", err)
	}
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	conn.Close(ctx)
	if err != nil {
		return nil, err
	}

	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		return nil, err
	}
	cfg.AfterConnect = pgxvec.RegisterTypes
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if _, err := pool.Exec(ctx, schema); err != nil {
		pool.Close()
		return nil, err
	}
	return &Store{pool: pool}, nil
}

func (s *Store) Close() { s.pool.Close() }

// ReplaceFile swaps out every chunk of one file in a single transaction, so
// re-running ingest never leaves duplicates or half-indexed files.
func (s *Store) ReplaceFile(ctx context.Context, project, file string, chunks []chunk.Chunk, vecs [][]float32) error {
	if len(chunks) != len(vecs) {
		return fmt.Errorf("%d chunks but %d vectors", len(chunks), len(vecs))
	}
	return pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, "DELETE FROM chunks WHERE file = $1", file); err != nil {
			return err
		}
		batch := &pgx.Batch{}
		for i, c := range chunks {
			batch.Queue(`INSERT INTO chunks (project, file, kind, name, start_line, end_line, content, embedding)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
				project, file, c.Kind, c.Name, c.StartLine, c.EndLine, c.Content, pgvector.NewVector(vecs[i]))
		}
		return tx.SendBatch(ctx, batch).Close()
	})
}

type Result struct {
	chunk.Chunk
	Project string
	Score   float64 // cosine similarity, 1 = same meaning
}

// Search returns the k chunks closest in meaning to the query vector.
// `<=>` is pgvector's cosine distance (1 - cosine similarity), so ordering by
// it ascending puts the most similar chunks first.
func (s *Store) Search(ctx context.Context, query []float32, k int) ([]Result, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT project, file, kind, name, start_line, end_line, content, 1 - (embedding <=> $1) AS score
		FROM chunks
		ORDER BY embedding <=> $1
		LIMIT $2`, pgvector.NewVector(query), k)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(row pgx.CollectableRow) (Result, error) {
		var r Result
		err := row.Scan(&r.Project, &r.File, &r.Kind, &r.Name, &r.StartLine, &r.EndLine, &r.Content, &r.Score)
		return r, err
	})
}

// Count returns how many chunks are indexed.
func (s *Store) Count(ctx context.Context) (int, error) {
	var n int
	err := s.pool.QueryRow(ctx, "SELECT count(*) FROM chunks").Scan(&n)
	return n, err
}
