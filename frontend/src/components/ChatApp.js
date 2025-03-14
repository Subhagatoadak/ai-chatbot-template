import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, Card, CardHeader, CardContent, CardActions, TextField, IconButton, Avatar, Typography 
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import ReactMarkdown from 'react-markdown';
import './ChatApp.css';

// Import notification sound from local folder
import notificationSound from '../sounds/notification.mp3';

export default function ChatApp() {
  const [messages, setMessages] = useState([]); // start with no messages
  const [inputValue, setInputValue] = useState('');
  const [apiLoading, setApiLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const messageEndRef = useRef(null);
  const recognitionRef = useRef(null);

  // Auto-scroll to the bottom whenever messages update
  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Setup SpeechRecognition for voice input (if supported)
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(transcript);
      };
      recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        setListening(false);
      };
      recognition.onend = () => {
        setListening(false);
      };
      recognitionRef.current = recognition;
    }
  }, []);

  const handleVoiceInput = () => {
    if (recognitionRef.current) {
      if (!listening) {
        setListening(true);
        recognitionRef.current.start();
      } else {
        recognitionRef.current.stop();
        setListening(false);
      }
    }
  };

  // Play notification sound when bot response is received
  const playNotificationSound = () => {
    const audio = new Audio(notificationSound);
    audio.play();
  };

  // Send the user's message and call the backend API to get the bot response
  const handleSend = async () => {
    if (!inputValue.trim()) return;
    const userMessage = {
      id: Date.now(),
      sender: 'Subho',
      text: inputValue.trim()
    };
    // Add user's message and clear input
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setApiLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.text })
      });
      const data = await response.json();
      const botMessage = {
        id: Date.now() + 1,
        sender: 'Omnis',
        text: data.answer || 'No answer provided.'
      };
      setMessages((prev) => [...prev, botMessage]);
      // Play notification sound when bot response is received
      playNotificationSound();
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 2,
        sender: 'Omnis',
        text: 'Error communicating with the chatbot.'
      };
      setMessages((prev) => [...prev, errorMessage]);
      playNotificationSound();
    } finally {
      setApiLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // A simple loading indicator with jumping dots
  const LoadingDots = () => (
    <span className="loading-dots">
      <span>.</span>
      <span>.</span>
      <span>.</span>
    </span>
  );

  return (
    <Box className="chat-container">
      <Card className="chat-card">
        <CardHeader 
          avatar={<Avatar src="https://i.pravatar.cc/40?img=60" />} 
          title={<Typography variant="h6">Omnis Chatbot</Typography>}
          subheader={<Typography variant="caption">Subhagato Adak</Typography>}
          className="chat-header"
        />
        <CardContent className="chat-content">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.sender === 'Subho' ? 'message-right' : 'message-left'}`}>
              {msg.sender === 'Omnis' ? (
                // Render bot response as markdown using react-markdown
                <ReactMarkdown className="bot-message">{msg.text}</ReactMarkdown>
              ) : (
                <Typography variant="body1">{msg.text}</Typography>
              )}
            </div>
          ))}
          {apiLoading && (
            <div className="message message-left">
              <Typography variant="body1">
                <LoadingDots />
              </Typography>
            </div>
          )}
          <div ref={messageEndRef} />
        </CardContent>
        <CardActions className="chat-actions">
          <TextField 
            fullWidth
            multiline
            minRows={1}
            maxRows={4}
            variant="outlined"
            placeholder="Type your message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <IconButton color="primary" onClick={handleVoiceInput}>
            <MicIcon color={listening ? "error" : "inherit"} />
          </IconButton>
          <IconButton color="primary" onClick={handleSend}>
            <SendIcon />
          </IconButton>
        </CardActions>
      </Card>
    </Box>
  );
}
