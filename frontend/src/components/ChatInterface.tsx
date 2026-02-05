import { useState, useRef, useEffect, useCallback, FormEvent } from 'react';
import { Send, Loader2, Sparkles } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatInterfaceProps {
  sessionId: string;
  suggestedPrompts: string[];
  onPromptsUsed?: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL;

export function ChatInterface({ sessionId, suggestedPrompts, onPromptsUsed }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [showPrompts, setShowPrompts] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: "Hi! I'm your financial coach. I've analyzed your transactions and I'm ready to help. What would you like to know about your spending?",
        timestamp: new Date()
      }]);
    }
  }, []);

  const sendMessage = useCallback(async (messageText: string) => {
    if (!messageText.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: messageText.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setShowPrompts(false);

    const assistantId = `assistant-${Date.now()}`;
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    }]);

    try {
      const response = await fetch(
        `${API_BASE}/chat/${sessionId}/sync`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: messageText.trim(),
            conversation_id: conversationId
          })
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      setMessages(prev => prev.map(msg =>
        msg.id === assistantId
          ? { ...msg, content: data.message || "I'm not sure how to respond to that." }
          : msg
      ));

      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId
          ? { ...msg, content: "I'm sorry, I encountered an error. Please try again." }
          : msg
      ));
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [sessionId, conversationId, isLoading]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handlePromptClick = (prompt: string) => {
    sendMessage(prompt);
    onPromptsUsed?.();
  };

  return (
    <div className="flex flex-col h-full min-h-0 bg-surface rounded-2xl shadow-sm border border-border/60 overflow-hidden">

      {/* Header */}
      <div className="flex-shrink-0 px-5 py-4 border-b border-border/60 bg-gradient-to-r from-[#E8F4F8] to-[#F5E6F3]">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-[#7C3AED]" />
          <h2 className="text-base font-semibold text-textPrimary">Chat with Dinera</h2>
        </div>
        <p className="text-xs text-muted mt-0.5">Your AI financial assistant</p>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-2.5 ${message.role === 'user'
              ? 'bg-[#A8D8EA] text-[#1a365d]'
              : 'bg-[#F5E6F3] text-textPrimary'
              }`}>
              <div className="text-sm whitespace-pre-wrap leading-relaxed">
                {message.content || (
                  <span className="flex items-center space-x-2 text-muted">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Thinking...</span>
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}

        {showPrompts && suggestedPrompts.length > 0 && messages.length <= 1 && (
          <div className="pt-3">
            <p className="text-xs text-muted mb-2">Try asking:</p>
            <div className="flex flex-wrap gap-1.5">
              {suggestedPrompts.slice(0, 4).map((prompt, index) => (
                <button
                  key={index}
                  onClick={() => handlePromptClick(prompt)}
                  className="text-xs px-3 py-1.5 rounded-full bg-[#E8F4F8] text-[#1a365d] hover:bg-[#A8D8EA] border border-[#A8D8EA]/50"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex-shrink-0 p-3 border-t border-border/60 bg-surface">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your spending..."
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 bg-bg rounded-xl text-sm focus:outline-none"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-2.5 rounded-xl bg-[#A8D8EA] hover:bg-[#8BC4D9]"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </form>
    </div>
  );
}
























// import { useState, useRef, useEffect, useCallback, FormEvent } from 'react';
// import { Send, Loader2, Sparkles } from 'lucide-react';

// interface Message {
//   id: string;
//   role: 'user' | 'assistant';
//   content: string;
//   timestamp: Date;
// }

// interface ChatInterfaceProps {
//   sessionId: string;
//   suggestedPrompts: string[];
//   onPromptsUsed?: () => void;
// }

// export function ChatInterface({ sessionId, suggestedPrompts, onPromptsUsed }: ChatInterfaceProps) {
//   const [messages, setMessages] = useState<Message[]>([]);
//   const [input, setInput] = useState('');
//   const [isLoading, setIsLoading] = useState(false);
//   const [conversationId, setConversationId] = useState<string | null>(null);
//   const [showPrompts, setShowPrompts] = useState(true);

//   const messagesEndRef = useRef<HTMLDivElement>(null);
//   const inputRef = useRef<HTMLInputElement>(null);

//   // Auto-scroll to bottom when messages change
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [messages]);

//   // Add welcome message on mount
//   useEffect(() => {
//     if (messages.length === 0) {
//       setMessages([{
//         id: 'welcome',
//         role: 'assistant',
//         content: "Hi! I'm your financial coach. I've analyzed your transactions and I'm ready to help. What would you like to know about your spending?",
//         timestamp: new Date()
//       }]);
//     }
//   }, []);

//   const sendMessage = useCallback(async (messageText: string) => {
//     if (!messageText.trim() || isLoading) return;

//     const userMessage: Message = {
//       id: `user-${Date.now()}`,
//       role: 'user',
//       content: messageText.trim(),
//       timestamp: new Date()
//     };

//     setMessages(prev => [...prev, userMessage]);
//     setInput('');
//     setIsLoading(true);
//     setShowPrompts(false);

//     // Create placeholder for assistant response
//     const assistantId = `assistant-${Date.now()}`;
//     setMessages(prev => [...prev, {
//       id: assistantId,
//       role: 'assistant',
//       content: '',
//       timestamp: new Date()
//     }]);

//     try {
//       // Use sync endpoint - simpler and returns conversation_id
//       const response = await fetch(`http://localhost:8000/chat/${sessionId}/sync`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           message: messageText.trim(),
//           conversation_id: conversationId
//         })
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP ${response.status}`);
//       }

//       const data = await response.json();

//       // Update message with response
//       setMessages(prev => prev.map(msg =>
//         msg.id === assistantId
//           ? { ...msg, content: data.message || "I'm not sure how to respond to that." }
//           : msg
//       ));

//       // Store conversation ID for context
//       if (data.conversation_id) {
//         setConversationId(data.conversation_id);
//       }

//     } catch (error) {
//       console.error('Chat error:', error);
//       setMessages(prev => prev.map(msg =>
//         msg.id === assistantId
//           ? { ...msg, content: "I'm sorry, I encountered an error. Please try again." }
//           : msg
//       ));
//     } finally {
//       setIsLoading(false);
//       inputRef.current?.focus();
//     }
//   }, [sessionId, conversationId, isLoading]);

//   const handleSubmit = (e: FormEvent) => {
//     e.preventDefault();
//     sendMessage(input);
//   };

//   const handlePromptClick = (prompt: string) => {
//     sendMessage(prompt);
//     onPromptsUsed?.();
//   };

//   return (
//     <div className="flex flex-col h-full min-h-0 bg-surface rounded-2xl shadow-sm border border-border/60 overflow-hidden">
//       {/* Header - fixed at top with subtle pastel gradient */}
//       <div className="flex-shrink-0 px-5 py-4 border-b border-border/60 bg-gradient-to-r from-[#E8F4F8] to-[#F5E6F3]">
//         <div className="flex items-center space-x-2">
//           <Sparkles className="w-5 h-5 text-[#7C3AED]" />
//           <h2 className="text-base font-semibold text-textPrimary">Chat with Dinera</h2>
//         </div>
//         <p className="text-xs text-muted mt-0.5">Your AI financial assistant</p>
//       </div>

//       {/* Messages - scrollable area with fixed height */}
//       <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
//         {messages.map((message) => (
//           <div
//             key={message.id}
//             className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
//           >
//             <div
//               className={`max-w-[85%] rounded-2xl px-4 py-2.5 ${message.role === 'user'
//                 ? 'bg-[#A8D8EA] text-[#1a365d]'
//                 : 'bg-[#F5E6F3] text-textPrimary'
//                 }`}
//             >
//               <div className="text-sm whitespace-pre-wrap leading-relaxed">
//                 {message.content || (
//                   <span className="flex items-center space-x-2 text-muted">
//                     <Loader2 className="w-4 h-4 animate-spin" />
//                     <span>Thinking...</span>
//                   </span>
//                 )}
//               </div>
//             </div>
//           </div>
//         ))}

//         {/* Suggested Prompts */}
//         {showPrompts && suggestedPrompts.length > 0 && messages.length <= 1 && (
//           <div className="pt-3">
//             <p className="text-xs text-muted mb-2">Try asking:</p>
//             <div className="flex flex-wrap gap-1.5">
//               {suggestedPrompts.slice(0, 4).map((prompt, index) => (
//                 <button
//                   key={index}
//                   onClick={() => handlePromptClick(prompt)}
//                   className="text-xs px-3 py-1.5 rounded-full transition-all
//                            bg-[#E8F4F8] text-[#1a365d] hover:bg-[#A8D8EA]
//                            border border-[#A8D8EA]/50"
//                 >
//                   {prompt}
//                 </button>
//               ))}
//             </div>
//           </div>
//         )}

//         <div ref={messagesEndRef} />
//       </div>

//       {/* Input - sticky at bottom */}
//       <form onSubmit={handleSubmit} className="flex-shrink-0 p-3 border-t border-border/60 bg-surface">
//         <div className="flex items-center gap-2">
//           <input
//             ref={inputRef}
//             type="text"
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             placeholder="Ask about your spending..."
//             disabled={isLoading}
//             className="flex-1 px-4 py-2.5 bg-bg rounded-xl
//                      text-sm text-textPrimary placeholder-muted
//                      focus:outline-none focus:ring-1 focus:ring-accent/30
//                      disabled:opacity-50"
//           />
//           <button
//             type="submit"
//             disabled={!input.trim() || isLoading}
//             className={`p-2.5 rounded-xl transition-all ${input.trim() && !isLoading
//               ? 'bg-[#A8D8EA] text-[#1a365d] hover:bg-[#8BC4D9]'
//               : 'bg-bg text-muted cursor-not-allowed'
//               }`}
//           >
//             {isLoading ? (
//               <Loader2 className="w-4 h-4 animate-spin" />
//             ) : (
//               <Send className="w-4 h-4" />
//             )}
//           </button>
//         </div>
//       </form>
//     </div>
//   );
// }
