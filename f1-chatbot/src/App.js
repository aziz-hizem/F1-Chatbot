import React, { useState } from "react";
import axios from "axios"; // Import axios
import "./App.css"; // Optional styling

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;
  
    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
  
    try {
      console.log("Sending request...");
      const response = await axios.post(
        "http://127.0.0.1:8000/predict_intent/",
        { question: input },
        {
          headers: {
            "Content-Type": "application/json",  // Ensure the content type is correct
          },
        }
      );
  
      const botMessage = { sender: "bot", text: response.data.response };
      setMessages([...messages, userMessage, botMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
      const errorMessage = { sender: "bot", text: "Oops! Something went wrong." };
      setMessages([...messages, userMessage, errorMessage]);
    }
  
    setInput("");
  };
  
  return (
    <div className="chat-container">
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={msg.sender}>
            {msg.text}
          </div>
        ))}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me something about F1..."
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;
