import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";
import f1Logo from './assets/f1-logo.png';  // Import the F1 logo image

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [theme, setTheme] = useState("general f1"); // Default theme
  const chatBoxRef = useRef(null);  // Create a ref for the chat box

  // Theme data
  const themes = [
    { name: "General F1", color: "#09a638",logo: "generalf1_logo.png", background: "f1_bg.jpg" },
    { name: "Ferrari", color: "#ff2800",logo: "ferrari_logo.png", background: "ferrari_bg.jpg" },
    { name: "Mercedes", color: "#00d2be",logo : "mercedes_logo.png", background: "mercedes_bg.jpg" },
    { name: "Red Bull", color: "#0600ef", logo: "redbull_logo.png",background: "redbull_bg.jpg" },
    { name: "Mclaren", color: "#ff8700", logo: "mclaren_logo.png", background: "mclaren_bg.jpg" },
    { name: "Aston Martin", color: "#006f62", logo : "astonmartin_logo.png", background: "astonmartin_bg.jpg" },
    { name: "Alpine", color: "#0090ff",logo: "alpine_logo.png", background: "alpine_bg.jpg" },
    { name: "Williams", color: "#005aff",logo: "williams_logo.png", background: "williams_bg.jpg" },
    { name: "RB", color: "#2b4562",logo: "rb_logo.png", background: "rb_bg.jpg" },
    { name: "Kick Sauber", color: "#900000",logo: "kicksauber_logo.png", background: "kicksauber_bg.jpg" },
    { name: "Haas", color: "#ffffff",logo: "haas_logo.png", background: "haas_bg.jpg" },
  ];

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
    setInput("");
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

    
    setIsTyping(false);
  };

  const cancelMessage = () => {
    setInput("");  // Clear the input field
  };

  // Change theme
  const changeTheme = (selectedTheme) => {
    setTheme(selectedTheme.toLowerCase().replace(/\s+/g, '')); // Normalize theme name
  };

  // Find the selected theme
  const selectedTheme = themes.find(t => t.name.toLowerCase().replace(/\s+/g, '') === theme);

  // Debug theme and background image
  useEffect(() => {
    console.log("Selected Theme:", theme);
    console.log("Selected Theme Data:", selectedTheme);
  }, [theme]);

  return (
    <div className="app-container">
      {/* Left Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <img src={f1Logo} alt="F1 logo" className="f1-logo" />
        </div>
        <div className="theme-selection">
          <h3>Select Theme</h3>
          <ul>
            {themes.map((t, index) => (
              <li
                key={index}
                onClick={() => changeTheme(t.name)}
                className={theme === t.name.toLowerCase().replace(/\s+/g, '') ? "active" : ""}
              >
                <img 
                src={`${process.env.PUBLIC_URL}/assets/${t.logo}`} 
                alt={t.name} 
                style={{ width: '50px', height: '30px', objectFit: 'contain' }} // Fixed size
               />
              </li>
            ))}
          </ul>
        </div>
        <div className="by_hizem">
          <h3>By Aziz Hizem</h3>
        </div>
      </div>

      {/* Right Chat Container */}
      <div
        className="chat-container"
        style={{
          backgroundImage: selectedTheme ? `url(${process.env.PUBLIC_URL}/assets/${selectedTheme.background})` : "none",
          backgroundSize: "87%", // Ensures full coverage
          backgroundPosition: "right", // Aligns it properly
          backgroundRepeat: "no-repeat",
          backgroundAttachment: "fixed", // Prevents scrolling issues
          backgroundColor: "rgba(0, 0, 0, 0.7)", // Ensures text readability
        }}
      >
        <div className="chat-header">
          Ask me about <img src={f1Logo} alt="F1 logo" className="f1-logo" /> !
        </div>
        <div className="chat-box" ref={chatBoxRef}>
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
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button className="cancel" onClick={cancelMessage}>Cancel</button>
          <button
            onClick={sendMessage}
            style={{ backgroundColor: selectedTheme ? selectedTheme.color : "#FFFFFF",
              color: 'white'
             }}
            
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;