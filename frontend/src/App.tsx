import './App.css';
import ChatContainer from './components/ChatContainer';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Chat Bot</h1>
      </header>
      <main className="app-main">
        <ChatContainer />
      </main>
    </div>
  );
}

export default App;
