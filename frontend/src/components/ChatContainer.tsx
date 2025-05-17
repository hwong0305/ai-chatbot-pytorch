import React, { useState } from 'react';
import './ChatContainer.css';
import ChatInput from './ChatInput';
import ChatMessage from './ChatMessage';

interface Message {
  text: string;
  isBot: boolean;
  timestamp: Date;
}

const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      text: 'Hello! How can I help you today?',
      isBot: true,
      timestamp: new Date(),
    },
  ]);

  const handleSendMessage = async (message: string) => {
    try {
      if (message.trim() === '') return;

      // Add user message
      setMessages(prev => [
        ...prev,
        {
          text: message,
          isBot: false,
          timestamp: new Date(),
        },
      ]);

      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      const response = await resp.text();

      setMessages(prev => [
        ...prev,
        {
          text: response.replace(/"/g, ''),
          isBot: true,
          timestamp: new Date(),
        },
      ]);
    } catch (err) {
      console.error(err);
      alert('An error occurred. Please try again later.');
    }
  };

  return (
    <div className="chat-container">
      <div className="messages-container">
        {messages.map((message, index) => (
          <ChatMessage
            key={index}
            text={message.text}
            isBot={message.isBot}
            timestamp={message.timestamp}
          />
        ))}
      </div>
      <ChatInput onSendMessage={handleSendMessage} />
    </div>
  );
};

export default ChatContainer;
