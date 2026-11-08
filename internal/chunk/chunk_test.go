package chunk

import (
	"strings"
	"testing"
)

const goSrc = `package server

import "net/http"

// MaxConns limits open connections.
const MaxConns = 100

type Server struct {
	addr string
}

// Handle serves one request.
func (s *Server) Handle(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("ok"))
}

func New(addr string) *Server {
	return &Server{addr: addr}
}
`

func TestGo(t *testing.T) {
	chunks, err := Go([]byte(goSrc))
	if err != nil {
		t.Fatal(err)
	}

	want := []struct {
		kind, name string
		start, end int
	}{
		{"const", "MaxConns", 5, 6}, // doc comment included, import skipped
		{"type", "Server", 8, 10},
		{"func", "(*Server).Handle", 12, 15},
		{"func", "New", 17, 19},
	}
	if len(chunks) != len(want) {
		t.Fatalf("got %d chunks, want %d: %+v", len(chunks), len(want), chunks)
	}
	for i, w := range want {
		c := chunks[i]
		if c.Kind != w.kind || c.Name != w.name || c.StartLine != w.start || c.EndLine != w.end {
			t.Errorf("chunk %d = %s %q lines %d-%d, want %s %q lines %d-%d",
				i, c.Kind, c.Name, c.StartLine, c.EndLine, w.kind, w.name, w.start, w.end)
		}
	}
	if !strings.HasPrefix(chunks[2].Content, "// Handle serves one request.") {
		t.Errorf("method chunk should start with its doc comment, got %q", chunks[2].Content)
	}
}

func TestMarkdown(t *testing.T) {
	src := "# Title\nintro\n\n## Setup\nrun it\n```bash\n# not a heading\n```\n## Usage\ncall it"
	chunks := Markdown(src)

	want := []struct {
		name       string
		start, end int
	}{
		{"Title", 1, 3},
		{"Setup", 4, 8},
		{"Usage", 9, 10},
	}
	if len(chunks) != len(want) {
		t.Fatalf("got %d chunks, want %d: %+v", len(chunks), len(want), chunks)
	}
	for i, w := range want {
		c := chunks[i]
		if c.Name != w.name || c.StartLine != w.start || c.EndLine != w.end {
			t.Errorf("chunk %d = %q lines %d-%d, want %q lines %d-%d",
				i, c.Name, c.StartLine, c.EndLine, w.name, w.start, w.end)
		}
	}
}

func TestSplitLarge(t *testing.T) {
	// 200 lines of 50 chars = ~10,000 chars, so it must split.
	line := strings.Repeat("x", 49)
	var lines []string
	for range 200 {
		lines = append(lines, line)
	}
	c := Chunk{Kind: "func", Name: "Big", StartLine: 10, EndLine: 209, Content: strings.Join(lines, "\n")}

	pieces := splitLarge(c)
	if len(pieces) < 2 {
		t.Fatalf("expected a split, got %d piece", len(pieces))
	}
	if pieces[0].StartLine != 10 {
		t.Errorf("first piece starts at %d, want 10", pieces[0].StartLine)
	}
	if last := pieces[len(pieces)-1]; last.EndLine != 209 {
		t.Errorf("last piece ends at %d, want 209", last.EndLine)
	}
	for i, p := range pieces {
		if len(p.Content) > MaxChars {
			t.Errorf("piece %d has %d chars, over MaxChars", i, len(p.Content))
		}
		if i > 0 && p.StartLine != pieces[i-1].EndLine-OverlapLines+1 {
			t.Errorf("piece %d starts at %d, want %d lines of overlap with previous (ends %d)",
				i, p.StartLine, OverlapLines, pieces[i-1].EndLine)
		}
	}
}
