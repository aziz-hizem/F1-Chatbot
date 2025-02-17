import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";
import f1Logo from './assets/f1-logo.png';  // Import the F1 logo image

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const chatBoxRef = useRef(null);  // Create a ref for the chat box

  // Scroll to the bottom whenever messages change
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);  // Trigger this effect whenever messages change
  
  // Typing animation function
  const typeMessage = (message) => {
    return new Promise((resolve) => {
      let i = 0;
      const interval = setInterval(() => {
        setMessages((prevMessages) => {
          const lastMessage = prevMessages[prevMessages.length - 1];
          if (lastMessage.sender === "bot" && i < message.length) {
            lastMessage.text = message.slice(0, i + 1);
            return [...prevMessages.slice(0, -1), lastMessage];
          }
          clearInterval(interval);
          resolve();
          return prevMessages;
        });
        i++;
      }, 50); // Typing speed (ms)
    });
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);

    // Set bot message to indicate typing
    const botMessage = { sender: "bot", text: "" }; // Empty message initially
    setMessages([...messages, userMessage, botMessage]);

    // Start typing animation after the user sends a message
    setIsTyping(true);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict_intent/", {
        question: input,
      });

      // Typing effect for bot's response
      await typeMessage(response.data.response);
    } catch (error) {
      console.error("Error fetching response:", error);
      const errorMessage = { sender: "bot", text: "Oops! Something went wrong." };
      setMessages([...messages, userMessage, errorMessage]);
    }

    setInput("");
    setIsTyping(false);
  };

  const cancelMessage = () => {
    setInput("");  // Clear the input field
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        {/* Display the F1 logo instead of the text "F1" */}
        Ask me about <img src={f1Logo} alt="F1 logo" className="f1-logo" /> !
      </div>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={msg.sender}>
            {msg.text}
          </div>
        ))}
        {/* Add a typing indicator while bot is typing */}
        {isTyping && <div className="typing-indicator">...</div>}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me something about Formula 1..."
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button className="cancel" onClick={cancelMessage}>Cancel</button>
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;





