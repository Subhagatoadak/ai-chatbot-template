import React, { useState } from 'react';

function Chatbot() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const sendQuestion = async () => {
    if (!question) return;
    setLoading(true);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setResponse(data.answer);
    } catch (error) {
      setResponse('Error communicating with the chatbot.');
    }
    setLoading(false);
  };

  return (
    <div>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows="4"
        cols="50"
        placeholder="Ask your question..."
      />
      <br />
      <button onClick={sendQuestion} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>
      <div>
        <h3>Response:</h3>
        <p>{response}</p>
      </div>
    </div>
  );
}

export default Chatbot;
