// Package vector holds the similarity math behind retrieval.
package vector

import "math"

// Cosine returns the cosine similarity of a and b: 1 means same direction
// (same meaning), 0 means unrelated. Panics if lengths differ.
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("vector: length mismatch")
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
