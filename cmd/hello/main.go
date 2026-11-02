package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/simpleshaikh/askmycode/internal/llm"
)

func main() {
	question := "In three bullet points, what is RAG (retrieval-augmented generation)?"
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}

	// The system prompt sets the model's role and rules for the whole chat.
	// Try changing it and see how the answer changes.
	system := "You are a concise tutor for an experienced Go backend developer."

	msg, err := llm.New().Ask(context.Background(), system, question, os.Stdout)
	if err != nil {
		log.Fatal(err)
	}

	// Tokens are what you pay for. Watch how they change with longer prompts.
	fmt.Printf("\n\n[tokens] input=%d output=%d stop=%s\n",
		msg.Usage.InputTokens, msg.Usage.OutputTokens, msg.StopReason)
}
