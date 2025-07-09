// /Users/atishaymalik/Desktop/ungli/UNGLI/front_end/src/pages/Chat.tsx
import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, User, ArrowRight, Loader2, Download } from "lucide-react";
import { Link, useParams, useNavigate } from "react-router-dom";
import { chatService, Chat, ConversationTurn, SendMessageRequest, SendMessageSuccessResponse } from "@/services/chatService";
import { authService } from "@/services/authService";
import { useToast } from "@/hooks/use-toast";

// Helper function to download chat as CSV (this can stay outside or inside, but it's not a Hook)
const downloadChatAsCsv = (messages: ConversationTurn[], chatTitle: string | null) => {
    if (!messages || messages.length === 0) {
        alert("No messages to download for the current chat.");
        return;
    }

    const sanitizedTitle = (chatTitle || 'chat_export').replace(/[^a-z0-9_.-]/gi, '_');
    const filename = `${sanitizedTitle}_${new Date().toISOString().slice(0,10)}.csv`;

    let csvContent = "Sender,Timestamp,Content\n";

    messages.forEach(turn => {
        const sender = turn.role;
        // Correctly get content for CSV export based on role
        const content = turn.role === 'user' ? turn.answer : turn.question; // User's content is in 'answer', bot's in 'question'
        const timestamp = new Date(turn.timestamp).toLocaleString();
        const escapedContent = `"${content.replace(/"/g, '""')}"`;

        csvContent += `${sender},"${timestamp}",${escapedContent}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');

    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } else {
        window.open('data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent));
    }
};


const ChatPage = () => {
  const { chatId: routeChatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();

  // ⭐ ALL HOOKS MUST BE DECLARED UNCONDITIONALLY AT THE TOP LEVEL
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);
  const [isUserAuthenticated, setIsUserAuthenticated] = useState(false);

  const [messages, setMessages] = useState<ConversationTurn[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(routeChatId || null);

  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [isChatsLoading, setIsChatsLoading] = useState(false);
  const [isMessagesLoading, setIsMessagesLoading] = useState(false);

  const [responseStartTime, setResponseStartTime] = useState<number | null>(null);
  const [averageResponseTime, setAverageResponseTime] = useState<number>(0);
  const { toast } = useToast();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);

  const [showSettings, setShowSettings] = useState(false);

  // ⭐ ADD THIS REF: Store the previously loaded chatId to prevent unnecessary reloads
  const prevLoadedChatId = useRef<string | null>(null);

  const currentChatTitle = useMemo(() => {
    return chats.find(chat => chat.id === currentChatId)?.title || null;
  }, [chats, currentChatId]);

  const usageStats = useMemo(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const messagesToday = messages.filter(turn => {
      const messageDate = new Date(turn.timestamp);
      return messageDate >= today;
    }).length;

    const activeChats = chats.filter(chat => {
      const lastActivity = new Date(chat.lastActivity);
      const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
      return lastActivity >= twentyFourHoursAgo;
    }).length;

    return {
      messagesToday,
      activeChats,
      responseTime: averageResponseTime > 0 ? `${averageResponseTime.toFixed(1)}s` : '0.0s'
    };
  }, [messages, chats, averageResponseTime]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const loadChatMessages = useCallback(async (chatIdToLoad: string) => {
    if (!authService.isAuthenticated()) {
        console.warn("loadChatMessages: Not authenticated, returning.");
        setMessages([]);
        return;
    }

    // ⭐ CRITICAL CHANGE: Only load messages if the chat ID has genuinely changed
    // or if the messages array is empty (meaning it's an initial load or reset)
    if (chatIdToLoad === prevLoadedChatId.current && messages.length > 0 && !isMessagesLoading) {
        console.log("Messages for this chat already loaded. Skipping re-fetch.");
        return; // Exit early
    }

    console.log('Loading messages for chat:', chatIdToLoad);
    setIsMessagesLoading(true);
    try {
      const chatMessages = await chatService.getChatMessages(chatIdToLoad);
      setMessages(chatMessages);
      prevLoadedChatId.current = chatIdToLoad; // Update the ref after successful load

      // We might not need initialLoadComplete logic here anymore,
      // as `scrollToBottom` is also called after message send
      scrollToBottom();

    } catch (error) {
      console.error("Error loading chat messages:", error);
      if (error instanceof Error && error.message !== "TokenExpired") {
        toast({
          title: "Error",
          description: "Failed to load messages for this chat.",
          variant: "destructive",
        });
      }
      setMessages([]); // Clear messages on error
      prevLoadedChatId.current = null; // Reset ref on error to allow retry
    } finally {
      setIsMessagesLoading(false);
    }
    // Added messages.length and isMessagesLoading to dependency array
    // to correctly trigger when message state is empty or loading state changes
  }, [toast, scrollToBottom, messages.length, isMessagesLoading]);


  // Effect for handling chat ID changes, including initial load from URL
  useEffect(() => {
    // Only run if authentication is verified (or loading is complete)
    if (!isLoadingAuth && isUserAuthenticated) {
       // If URL chat ID exists and differs from current state, update state
       // This will cause this useEffect to run again with the new currentChatId
       if (routeChatId && routeChatId !== currentChatId) {
           setCurrentChatId(routeChatId);
       }
       // If no current chat ID, but chats are loaded, default to the first one
       else if (!currentChatId && chats.length > 0) {
           setCurrentChatId(chats[0].id);
       }

       // If currentChatId is set (either from URL, default, or user click), load its messages
       if (currentChatId) {
           loadChatMessages(currentChatId);
       } else if (chats.length === 0 && !isChatsLoading) {
           // If no chats and not loading, maybe it's a fresh start, clear messages
           setMessages([]);
       }
    }
  }, [currentChatId, loadChatMessages, chats, routeChatId, isLoadingAuth, isUserAuthenticated, isChatsLoading]);


  // Effect to load user's chats on component mount (after auth check is complete)
  useEffect(() => {
    // This effect runs once after auth check
    if (!isLoadingAuth && isUserAuthenticated) {
        loadUserChats();
    }
  }, [isLoadingAuth, isUserAuthenticated]); // Depend on auth status

  const loadUserChats = async () => {
    if (!authService.isAuthenticated()) {
        console.warn("loadUserChats: Not authenticated, returning.");
        setIsChatsLoading(false);
        setChats([]);
        return;
    }

    console.log('Loading user chats...');
    setIsChatsLoading(true);
    try {
      let chatsToProcess = await chatService.getUserChats();

      if (chatsToProcess.length === 0) {
        console.log("No existing chats found. Creating a new one automatically.");
        const newChat = await createNewChat(undefined, true);
        if (newChat) {
          chatsToProcess = [newChat];
          setChats(chatsToProcess);
          setCurrentChatId(newChat.id);
        } else {
          toast({
            title: "Error",
            description: "Failed to create initial chat. Please try logging in again.",
            variant: "destructive",
          });
          setIsChatsLoading(false);
          return;
        }
      } else {
        setChats(chatsToProcess);
      }

      if (!currentChatId && chatsToProcess.length > 0) {
        console.log("Setting first chat as active by default:", chatsToProcess[0].id);
        setCurrentChatId(chatsToProcess[0].id);
      } else if (currentChatId && chatsToProcess.length > 0) {
        const foundChat = chatsToProcess.find(chat => chat.id === currentChatId);
        if (!foundChat) {
          console.warn(`Chat ID ${currentChatId} from URL not found. Defaulting to first chat.`);
          setCurrentChatId(chatsToProcess[0].id);
        }
      }
    } catch (error) {
      console.error("Error loading user chats:", error);
      if (error instanceof Error && error.message === "TokenExpired") {
        // Auth service should handle redirect, but ensure state is clear
        setChats([]);
      } else {
        toast({
          title: "Error",
          description: "Failed to load your chat history.",
          variant: "destructive",
        });
        setChats([]);
      }
    } finally {
      setIsChatsLoading(false);
    }
  };

  const createNewChat = async (defaultTitle?: string, isInitialLoad = false): Promise<Chat | null> => {
    if (!authService.isAuthenticated()) {
        console.warn("createNewChat: Not authenticated, returning.");
        return null;
    }

    let newChatTitle: string | undefined = defaultTitle;

    if (!isInitialLoad) {
      const promptResult = window.prompt("Enter a name for your new chat (Leave empty for 'ungli-untitled'):");
      if (promptResult === null) {
        return null;
      }
      newChatTitle = promptResult === "" ? undefined : promptResult;
    }

    console.log('Creating new chat with title:', newChatTitle || 'ungli-untitled');

    try {
      const newChat = await chatService.createChat(newChatTitle);

      if (newChat && newChat.id) {
        if (!isInitialLoad) {
            setChats(prev => [newChat, ...prev]);
            setCurrentChatId(newChat.id);
            setMessages([]);
            prevLoadedChatId.current = null; // Reset this so new chat forces loadChatMessages
            toast({
                title: "Chat Created!",
                description: `New chat "${newChat.title || 'ungli-untitled'}" started.`,
            });
        }
        return newChat;
      } else {
        toast({
          title: "Error",
          description: "Failed to create new chat or received invalid chat data.",
          variant: "destructive",
        });
        console.error("createNewChat: Failed to get valid new chat ID or data.");
        return null;
      }
    } catch (error) {
      console.error("Error creating chat:", error);
      if (error instanceof Error && error.message === "TokenExpired") {
        // Auth service handles redirect
      } else {
        toast({
          title: "Error",
          description: "Failed to create new chat",
          variant: "destructive",
        });
      }
      return null;
    }
  };

  const sendMessage = async () => {
    const content = newMessage.trim();
    if (!content || !currentChatId) {
        console.warn("Attempted to send empty message or no chat selected.");
        return;
    }

    if (!authService.isAuthenticated()) {
        console.warn("sendMessage: Not authenticated, returning.");
        return;
    }

    setIsSendingMessage(true);
    setNewMessage('');

    const tempUserTurn: ConversationTurn = {
      question: "", // User's message is the answer in ConversationTurn
      answer: content,
      timestamp: new Date().toISOString(),
      role: 'user',
    };

    const tempBotPlaceholder: ConversationTurn = {
      question: "Thinking...", // Bot's message is the question in ConversationTurn
      answer: "", // Bot's answer is expected to be empty based on your backend structure
      timestamp: new Date().toISOString(),
      role: 'assistant',
    };

    setMessages(prev => [...prev, tempUserTurn, tempBotPlaceholder]);
    scrollToBottom();

    setResponseStartTime(Date.now());

    try {
      const sendData: SendMessageRequest = { chatId: currentChatId, content: content };
      const apiResponse: SendMessageSuccessResponse = await chatService.sendMessage(sendData);

      if (apiResponse && apiResponse.message && apiResponse.message.question) {
          const actualBotTurn: ConversationTurn = {
              question: apiResponse.message.question, // This is where the bot's content is
              answer: "", // Ensure 'answer' is empty for assistant roles based on backend's structure
              timestamp: apiResponse.message.timestamp || new Date().toISOString(),
              role: 'assistant',
          };

          setMessages(prev => {
              const updatedMessages = [...prev];
              // Find the last placeholder to replace it
              const lastPlaceholderIndex = updatedMessages.findLastIndex(msg =>
                  msg.role === 'assistant' && msg.question === "Thinking..."
              );

              if (lastPlaceholderIndex !== -1) {
                  updatedMessages[lastPlaceholderIndex] = actualBotTurn;
              } else {
                  // Fallback: if placeholder not found (should ideally not happen), just add
                  updatedMessages.push(actualBotTurn);
              }
              return updatedMessages;
          });

          // Update the last message and activity for the current chat in the sidebar
          setChats(prevChats =>
              prevChats.map(chat =>
                  chat.id === currentChatId
                      ? { ...chat, lastMessage: content, lastActivity: actualBotTurn.timestamp } // Use the actual bot turn's timestamp
                      : chat
              )
          );

      } else {
          // ⭐ CRITICAL CHANGE: Removed loadChatMessages here.
          // This block now indicates an issue with the sendMessage response itself.
          console.error("Backend did not return a valid message in sendMessage response:", apiResponse);
          toast({
            title: "Error",
            description: "Bot response was invalid. Please try again.",
            variant: "destructive",
          });
          // Remove temp messages if bot response is invalid or missing
          setMessages(prev => {
            const filtered = prev.filter(msg =>
                !(msg.role === 'user' && msg.answer === content) &&
                !(msg.role === 'assistant' && msg.question === "Thinking...")
            );
            return filtered;
          });
      }

      if (responseStartTime) {
          const responseTime = (Date.now() - responseStartTime) / 1000;
          setAverageResponseTime(prev => prev === 0 ? responseTime : (prev + responseTime) / 2);
      }
      scrollToBottom();
    } catch (error) {
      console.error('Error sending message:', error);
      // Remove temporary messages on any send error
      setMessages(prev => {
        const filtered = prev.filter(msg =>
            !(msg.role === 'user' && msg.answer === content) &&
            !(msg.role === 'assistant' && msg.question === "Thinking...")
        );
        return filtered;
      });

      if (error instanceof Error && error.message === "TokenExpired") {
        // Auth service handles redirect
      } else {
        toast({
          title: "Error",
          description: "Failed to send message. Please try again.",
          variant: "destructive",
        });
      }
    } finally {
      setIsSendingMessage(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isSendingMessage && newMessage.trim() && currentChatId) {
      sendMessage();
    }
  };


  // ⭐ PRIMARY AUTHENTICATION EFFECT: Runs once on mount to determine auth status
  useEffect(() => {
    const checkAuthentication = () => {
      const authenticated = authService.isAuthenticated();
      setIsUserAuthenticated(authenticated);
      setIsLoadingAuth(false); // Auth check is complete

      if (!authenticated) {
        console.warn("ChatPage: User not authenticated. Redirecting to login.");
        navigate('/login');
      }
    };
    checkAuthentication();
  }, [navigate]); // navigate is stable

  // Now, your conditional rendering logic can use the state set by the hooks
  // and will not violate the Rules of Hooks because all hooks are always called.
  if (isLoadingAuth) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-2 text-lg text-gray-600">Verifying authentication...</span>
      </div>
    );
  }

  if (!isUserAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>Please Log In</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600 mb-4">You need to be logged in to access the chat.</p>
            <Link to="/login">
              <Button className="w-full">Go to Login</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  // --- Only render the main chat UI if authenticated ---
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-2">
            <MessageCircle className="h-6 w-6 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">UNGLI</span>
          </Link>
          <div className="flex items-center space-x-4">
            <Button variant="outline" size="sm" onClick={() => setShowSettings(true)}>
             Settings
            </Button>
            <Button variant="outline" size="sm" onClick={() => {
              authService.logout();
              navigate('/login');
            }}>
              <User className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </nav>

      <div className="flex h-[calc(100vh-73px)]">
        {/* Left Sidebar - Chat History */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Chat History</h2>
            <Button className="w-full mt-3" size="sm" onClick={() => createNewChat()}>
              + New Chat
            </Button>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-4 space-y-2">
              {isChatsLoading ? (
                <div className="flex justify-center items-center h-20">
                  <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                  <span className="ml-2 text-gray-600">Loading chats...</span>
                </div>
              ) : chats.length === 0 ? (
                <p className="text-center text-gray-500 text-sm mt-4">No conversations yet.</p>
              ) : (
                chats.map((chat) => (
                  <Card
                    key={chat.id}
                    className={`cursor-pointer hover:bg-gray-50 transition-colors border-l-4 ${
                      currentChatId === chat.id ? 'border-l-blue-500 bg-blue-50' : 'border-l-transparent'
                    }`}
                    onClick={() => setCurrentChatId(chat.id)}
                  >
                    <CardContent className="p-3">
                      <h3 className="font-medium text-sm text-gray-900 truncate">
                        {chat.title || 'Ungli-untitled Chat'}
                      </h3>
                      <p className="text-xs text-gray-500 mt-1 truncate">
                        {chat.lastMessage}
                      </p>
                      <p className="text-xs text-gray-400 mt-1">
                        {new Date(chat.lastActivity).toLocaleTimeString()}
                      </p>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </ScrollArea>
          {/* Bottom Buttons */}
          <div className="p-4 border-t border-gray-200 space-y-2">
            <Button className="w-full" variant="secondary" onClick={() => console.log("Deep Search Clicked")}>
              Deep Search
            </Button>
          </div>
        </div>

        {/* Center - Main Chat */}
        <div className="flex-1 flex flex-col">
          {/* Chat Messages */}
          <ScrollArea className="flex-1 p-6">
            <div className="space-y-4 max-w-4xl mx-auto" ref={messagesEndRef}>
              {isMessagesLoading ? (
                <div className="flex justify-center items-center h-full">
                  <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                  <span className="ml-2 text-lg text-gray-600">Loading messages...</span>
                </div>
              ) : messages.length === 0 && !isMessagesLoading && !isChatsLoading ? (
                 <p className="text-center text-gray-500 text-lg mt-20">
                    Start chatting! Type a message below.
                </p>
              ) : (
                messages.map((turn, index) => (
                  <div
                      key={turn.timestamp + '-' + index}
                      className={`flex ${turn.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl rounded-lg px-4 py-2 ${
                        turn.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-white border border-gray-200 text-gray-900'
                      }`}
                    >
                      {/* Display user's answer, bot's question */}
                      {turn.role === 'user' && turn.answer && <p className="text-sm">{turn.answer}</p>}
                      {turn.role === 'assistant' && turn.question && <p className="text-sm">{turn.question}</p>}

                      <p className={`text-xs mt-1 ${
                        turn.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                      }`}>
                        {new Date(turn.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>

          {/* Message Input */}
          <div className="border-t border-gray-200 p-6 bg-white">
            <div className="max-w-4xl mx-auto">
              <div className="flex space-x-4">
                <Input
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here..."
                  className="flex-1"
                  disabled={isSendingMessage || isMessagesLoading || isChatsLoading || !currentChatId}
                />
                <Button
                    onClick={sendMessage}
                    disabled={isSendingMessage || !newMessage.trim() || !currentChatId}
                >
                  {isSendingMessage ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <ArrowRight className="h-4 w-4" />
                  )}
                  {isSendingMessage ? 'Sending...' : 'Send'}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Sidebar - Chat Insights */}
        <div className="w-80 bg-white border-l border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Quick Starters</h2>
            <Button
              className="w-full mt-3"
              size="sm"
              onClick={() => downloadChatAsCsv(messages, currentChatTitle)}
              disabled={isMessagesLoading || !currentChatId || messages.length === 0}
            >
              <Download className="h-4 w-4 mr-2" />
              Download Chat (CSV)
            </Button>
          </div>
          <div className="flex-1 p-4 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Usage Stats</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Messages today:</span>
                  <span className="font-medium">{usageStats.messagesToday}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Active chats:</span>
                  <span className="font-medium">{usageStats.activeChats}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Response time:</span>
                  <span className="font-medium">{usageStats.responseTime}</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex justify-end">
          <div className="w-full max-w-md h-full bg-white shadow-xl p-6 overflow-y-auto">
            <div className="flex justify-between items-center border-b pb-4 mb-4">
              <h2 className="text-xl font-semibold">Settings</h2>
              <Button variant="ghost" onClick={() => setShowSettings(false)}>
                Close
              </Button>
            </div>

            {/* Settings Options */}
            <div className="space-y-6">
              {/* Profile */}
              <section>
                <h3 className="text-lg font-medium mb-2">Profile</h3>
                {authService.isAuthenticated() ? (
                    <>
                        {(() => {
                            const user = authService.getUser();
                            if (!user) return <p className="text-sm text-gray-600">User data not available.</p>;
                            return (
                                <>
                                    <p className="text-sm text-gray-600">
                                        Logged in as: <strong>{user.username || user.email || 'User'}</strong>
                                    </p>
                                    {user.profile?.name && (
                                        <p className="text-sm text-gray-600">Name: {user.profile.name}</p>
                                    )}
                                    <p className="text-sm text-gray-600">Manage your profile details like name and email.</p>
                                </>
                            );
                        })()}
                    </>
                ) : (
                    <p className="text-sm text-gray-600">Not logged in.</p>
                )}
                <Button variant="secondary" className="mt-2" onClick={() => console.log("Edit Profile clicked. Implement profile editing later.")}>Edit Profile</Button>
              </section>

              {/* Usage Statistics */}
              <section>
                <h3 className="text-lg font-medium mb-2">Usage Statistics</h3>
                <div className="text-sm text-gray-600">
                  <p>Messages Today: <strong>{usageStats.messagesToday}</strong></p>
                  <p>Active Chats: <strong>{usageStats.activeChats}</strong></p>
                  <p>Avg. Response Time: <strong>{usageStats.responseTime}</strong></p>
                </div>
              </section>

              {/* Payment & Account */}
              <section>
                <h3 className="text-lg font-medium mb-2">Payment & Account</h3>
                <p className="text-sm text-gray-600">Manage your billing and subscription.</p>
                <Button variant="secondary" className="mt-2">Manage Billing</Button>
              </section>

              {/* Security */}
              <section>
                <h3 className="text-lg font-medium mb-2">Security Settings</h3>
                <p className="text-sm text-gray-600">Update your password or enable 2FA.</p>
                <Button variant="secondary" className="mt-2">Update Security</Button>
              </section>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default ChatPage;