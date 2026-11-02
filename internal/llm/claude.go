// Package llm wraps the Claude API for the rest of the app.
package llm

import (
	"context"
	"fmt"
	"io"

	"github.com/anthropics/anthropic-sdk-go"
)

const DefaultModel = "claude-opus-5-5"

type Client struct {
	api   anthropic.Client
	model string
}

// New reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
func New() *Client {
	return &Client{api: anthropic.NewClient(), model: DefaultModel}
}

// Ask sends one question with a system prompt and streams the answer into w
// token by token. It returns the full message so callers can inspect usage.
func (c *Client) Ask(ctx context.Context, system, question string, w io.Writer) (*anthropic.Message, error) {
	stream := c.api.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:     c.model,
		MaxTokens: 64000,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(question)),
		},
		// Effort controls how hard the model thinks. "low" is plenty for Q&A
		// and keeps cost down; try "high" later and compare answers.
		OutputConfig: anthropic.OutputConfigParam{Effort: anthropic.OutputConfigEffortLow},
	})

	msg := anthropic.Message{}
	for stream.Next() {
		event := stream.Current()
		if err := msg.Accumulate(event); err != nil {
			return nil, err
		}
		if delta, ok := event.AsAny().(anthropic.ContentBlockDeltaEvent); ok {
			if text, ok := delta.Delta.AsAny().(anthropic.TextDelta); ok {
				fmt.Fprint(w, text.Text)
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	if msg.StopReason == anthropic.StopReasonRefusal {
		return &msg, fmt.Errorf("model declined (%s): %s", msg.StopDetails.Category, msg.StopDetails.Explanation)
	}
	return &msg, nil
}
