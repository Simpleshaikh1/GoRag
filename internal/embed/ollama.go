// Package embed turns text into vectors using a local Ollama server.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

const DefaultModel = "nomic-embed-text"

type Ollama struct {
	BaseURL string
	Model   string
	HTTP    *http.Client
}

func NewOllama() *Ollama {
	return &Ollama{BaseURL: "http://localhost:11434", Model: DefaultModel, HTTP: http.DefaultClient}
}

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// nomic-embed-text was trained with task prefixes: stored text and questions
// are marked differently, which makes question-to-document matching sharper.
const (
	documentPrefix = "search_document: "
	queryPrefix    = "search_query: "
)

// EmbedDocuments embeds text that will be stored and searched.
func (o *Ollama) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	prefixed := make([]string, len(texts))
	for i, t := range texts {
		prefixed[i] = documentPrefix + t
	}
	return o.Embed(ctx, prefixed)
}

// EmbedQuery embeds a user's question.
func (o *Ollama) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	vecs, err := o.Embed(ctx, []string{queryPrefix + text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// Embed returns one vector per input text, in the same order, with no prefix.
func (o *Ollama) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Model: o.Model, Input: texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling ollama (is it running?): %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned %s (did you run `ollama pull %s`?)", resp.Status, o.Model)
	}

	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("got %d embeddings for %d texts", len(out.Embeddings), len(texts))
	}
	return out.Embeddings, nil
}
