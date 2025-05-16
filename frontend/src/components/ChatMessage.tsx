import React from 'react';
import './ChatMessage.css';

interface ChatMessageProps {
  text: string;
  isBot: boolean;
  timestamp: Date;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ text, isBot, timestamp }) => {
  return (
    <div className={`chat-message ${isBot ? 'bot' : 'user'}`}>
      <div className="message-content">
        <p>{text}</p>
        <span className="timestamp">
          {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
};

export default ChatMessage;
