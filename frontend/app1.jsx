
import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim() || loading) {
      return;
    }

    const currentQuestion = question;

    setQuestion("");

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: currentQuestion,
      },
    ]);

    setLoading(true);

    try {
      const response = await fetch(
        "http://localhost:8000/ask",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: currentQuestion,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Request failed");
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Something went wrong while contacting the server.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="app">

      {/* Sidebar */}
      <aside className="sidebar">

        <div>
          <h2>📄 PDF AI</h2>

          <p className="sidebar-description">
            Ask questions about your PDF using
            Gemini and ChromaDB.
          </p>
        </div>

        <div className="document-card">
          <span>📕</span>

          <div>
            <strong>GRU.pdf</strong>
            <p>Knowledge source</p>
          </div>
        </div>

        <button
          className="clear-button"
          onClick={clearChat}
        >
          🗑 Clear conversation
        </button>

      </aside>


      {/* Main content */}
      <main className="chat-container">

        {/* Header */}
        <header className="header">

          <div>
            <h1>KAZI PDF Assistant</h1>

            <p>
              Ask anything about your document
            </p>
          </div>

          <div className="status">
            <span className="status-dot"></span>
            Gemini + Chroma
          </div>

        </header>


        {/* Messages */}
        <section className="messages">

          {messages.length === 0 && (
            <div className="welcome">

              <div className="welcome-icon">
                📚
              </div>

              <h2>
                Ask your PDF anything
              </h2>

              <p>
                I'll search the document and answer
                using the retrieved information.
              </p>

              <div className="suggestions">

                <button
                  onClick={() =>
                    setQuestion("What is GRU?")
                  }
                >
                  What is GRU?
                </button>

                <button
                  onClick={() =>
                    setQuestion(
                      "Explain the main concepts in the document."
                    )
                  }
                >
                  Explain the main concepts
                </button>

                <button
                  onClick={() =>
                    setQuestion(
                      "Give me a summary of the document."
                    )
                  }
                >
                  Summarize the document
                </button>

              </div>

            </div>
          )}


          {messages.map((message, index) => (

            <div
              key={index}
              className={`message ${
                message.role === "user"
                  ? "user-message"
                  : "assistant-message"
              }`}
            >

              <div className="avatar">

                {message.role === "user"
                  ? "👤"
                  : "🤖"}

              </div>

              <div className="message-content">

                <div className="message-name">

                  {message.role === "user"
                    ? "You"
                    : "PDF Assistant"}

                </div>

                <div className="message-text">

                  {message.content}

                </div>

              </div>

            </div>

          ))}


          {loading && (
            <div className="message assistant-message">

              <div className="avatar">
                🤖
              </div>

              <div className="message-content">

                <div className="message-name">
                  PDF Assistant
                </div>

                <div className="typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>

              </div>

            </div>
          )}

        </section>


        {/* Input */}
        <div className="input-area">

          <div className="input-wrapper">

            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Ask something about your PDF..."
              rows="1"
            />

            <button
              onClick={askQuestion}
              disabled={
                loading || !question.trim()
              }
              className="send-button"
            >
              ➤
            </button>

          </div>

          <p className="input-hint">
            Press Enter to send · Shift + Enter for a new line
          </p>

        </div>

      </main>
    </div>
  );
}

export default App;

