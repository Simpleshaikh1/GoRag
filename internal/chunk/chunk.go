// Package chunk splits source files into pieces small enough to embed and
// meaningful enough to retrieve on their own.
package chunk

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"strings"
)

// MaxChars caps a chunk's size. Bigger chunks get split by lines. About 4
// characters per token, so this is roughly 1,000 tokens: small enough for the
// embedding model and cheap to paste into a prompt.
const MaxChars = 4000

// OverlapLines is how many lines a split chunk repeats from the end of the
// previous piece, so code that straddles the cut appears whole in one of them.
const OverlapLines = 5

type Chunk struct {
	File      string // path relative to the indexed root, e.g. "JWT-AUTH/main.go"
	Kind      string // func, type, var, const, section, text
	Name      string // function, type or heading name; empty for plain text
	StartLine int    // 1-based, inclusive
	EndLine   int
	Content   string
}

// EmbedText is what gets embedded: the content plus a short header. The
// header lets a question like "auth middleware in JWT-AUTH" match on the file
// path and name, not just the code.
func (c Chunk) EmbedText() string {
	return fmt.Sprintf("File: %s\n%s %s\n\n%s", c.File, c.Kind, c.Name, c.Content)
}

// File picks a chunker by extension. Unsupported files return nil.
func File(path string, src []byte) []Chunk {
	var chunks []Chunk
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		var err error
		chunks, err = Go(src)
		if err != nil {
			// Broken Go files still have useful text in them.
			chunks = []Chunk{{Kind: "text", StartLine: 1, Content: string(src)}}
			chunks[0].EndLine = strings.Count(chunks[0].Content, "\n") + 1
		}
	case ".md":
		chunks = Markdown(string(src))
	default:
		return nil
	}

	var out []Chunk
	for _, c := range chunks {
		c.File = path
		out = append(out, splitLarge(c)...)
	}
	return out
}

// Go makes one chunk per top-level declaration (func, method, type, var,
// const), including its doc comment. Imports are skipped: they say little
// about what the code does.
func Go(src []byte) ([]Chunk, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var out []Chunk
	for _, decl := range f.Decls {
		start := decl.Pos()
		var kind, name string

		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Doc != nil {
				start = d.Doc.Pos()
			}
			kind, name = "func", funcName(d)
		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				continue
			}
			if d.Doc != nil {
				start = d.Doc.Pos()
			}
			kind, name = d.Tok.String(), genDeclName(d)
		default:
			continue
		}

		s, e := fset.Position(start), fset.Position(decl.End())
		out = append(out, Chunk{
			Kind:      kind,
			Name:      name,
			StartLine: s.Line,
			EndLine:   e.Line,
			Content:   string(src[s.Offset:e.Offset]),
		})
	}
	return out, nil
}

// funcName returns "Handle" for a function and "(*Server).Handle" for a method.
func funcName(d *ast.FuncDecl) string {
	if d.Recv == nil || len(d.Recv.List) == 0 {
		return d.Name.Name
	}
	return "(" + types.ExprString(d.Recv.List[0].Type) + ")." + d.Name.Name
}

// genDeclName lists the names a type/var/const block declares.
func genDeclName(d *ast.GenDecl) string {
	var names []string
	for _, spec := range d.Specs {
		switch s := spec.(type) {
		case *ast.TypeSpec:
			names = append(names, s.Name.Name)
		case *ast.ValueSpec:
			for _, n := range s.Names {
				names = append(names, n.Name)
			}
		}
	}
	if len(names) > 3 {
		names = append(names[:3], "...")
	}
	return strings.Join(names, ", ")
}

// Markdown makes one chunk per heading section. Lines starting with "#"
// inside ``` code fences are not headings.
func Markdown(src string) []Chunk {
	lines := strings.Split(src, "\n")
	var out []Chunk
	cur := Chunk{Kind: "section", StartLine: 1}
	var body []string
	inFence := false

	flush := func(endLine int) {
		text := strings.TrimSpace(strings.Join(body, "\n"))
		if text != "" {
			cur.EndLine = endLine
			cur.Content = strings.Join(body, "\n")
			out = append(out, cur)
		}
		body = nil
	}

	for i, line := range lines {
		lineNo := i + 1
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			inFence = !inFence
		}
		if !inFence && strings.HasPrefix(line, "#") {
			flush(lineNo - 1)
			cur = Chunk{Kind: "section", Name: strings.TrimSpace(strings.TrimLeft(line, "#")), StartLine: lineNo}
		}
		body = append(body, line)
	}
	flush(len(lines))
	return out
}

// splitLarge cuts a chunk over MaxChars into line-based windows that overlap
// by OverlapLines. Each piece keeps the original kind and name.
func splitLarge(c Chunk) []Chunk {
	if len(c.Content) <= MaxChars {
		return []Chunk{c}
	}

	lines := strings.Split(c.Content, "\n")
	var out []Chunk
	for start := 0; start < len(lines); {
		size, end := 0, start
		for end < len(lines) && (end == start || size+len(lines[end])+1 <= MaxChars) {
			size += len(lines[end]) + 1
			end++
		}

		piece := c
		piece.StartLine = c.StartLine + start
		piece.EndLine = c.StartLine + end - 1
		piece.Content = strings.Join(lines[start:end], "\n")
		out = append(out, piece)

		if end == len(lines) {
			break
		}
		// Step back for overlap, but always move forward at least one line.
		start = max(end-OverlapLines, start+1)
	}
	return out
}
