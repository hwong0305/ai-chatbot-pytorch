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

  const handleSendMessage = (message: string) => {
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

    // TODO: Add actual bot response logic here
    setTimeout(() => {
      setMessages(prev => [
        ...prev,
        {
          text: 'I received your message. This is a placeholder response.',
          isBot: true,
          timestamp: new Date(),
        },
      ]);
    }, 1000);
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
