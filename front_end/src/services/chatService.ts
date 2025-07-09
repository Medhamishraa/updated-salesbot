// /Users/atishaymalik/Desktop/ungli/UNGLI/front_end/src/services/chatService.ts

// import { authService } from './authService'; // ⭐ REMOVED: No longer directly using authService.fetchWithAuth for chat endpoints

// Determine the API base URL based on the environment.
const API_BASE_URL = import.meta.env.PROD
    ? "" // For Vercel/production, assuming same-origin proxy or Vercel config
    : "http://127.0.0.1:8006"; // <--- THIS IS YOUR FASTAPI BACKEND PORT

if (!import.meta.env.PROD) {
  console.warn(`[ChatService] Using development backend URL: ${API_BASE_URL}`);
}

// --- INTERFACES ---
// Export all interfaces directly from this file so Chat.tsx can import them.

// This represents the structure of a single "turn" in the transformed chat messages.
export interface ConversationTurn {
  question: string;
  answer: string;
  timestamp: string; // Keep as string for consistency with backend JSON
  role: 'user' | 'assistant';
}

// This represents the full response object from GET /api/chats/{chat_id}/messages
// as per your Pydantic model TransformedChatMessagesResponse.
export interface TransformedChatMessagesResponse {
  id: string; // This corresponds to your Pydantic `id: str = Field(alias="_id")`
  session_uuid: string;
  messages: ConversationTurn[]; // The array of conversation turns
}

// This is the payload structure for sending a new message.
export interface SendMessageRequest {
  chatId: string; // The chat ID to which the message belongs
  content: string; // The content of the message
}

// ⭐ UPDATED: This is the expected response from sending a message.
// Based on the error, your backend returns a `message` field, not `bot_message_instance`.
export interface SendMessageSuccessResponse {
  success: boolean;
  message: ConversationTurn; // ⭐ Changed from bot_message_instance to message
}

// This represents the Chat object returned by /api/chats and createChat.
export interface Chat {
  id: string; // Maps from backend `_id`
  participants: string[];
  lastMessage?: string; // Optional
  lastActivity: string; // Keep as string for consistency with backend JSON
  title: string;
}

class ChatService {
  // ⭐ REMOVED: No longer binding fetchWithAuth from authService
  // private fetchWithAuth = authService.fetchWithAuth.bind(authService);

  async getUserChats(): Promise<Chat[]> {
    try {
      // ⭐ MODIFIED: Use plain fetch for bypassed endpoint
      const response = await fetch(`${API_BASE_URL}/api/chats`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || 'Failed to fetch chats');
      }
      
      const result = await response.json(); // Expected: { success: true, chats: [...] }
      const chats = result.chats?.map((chat: any) => ({
        id: chat._id, // Map backend `_id` to frontend `id`
        participants: chat.participants,
        lastMessage: chat.lastMessage,
        lastActivity: chat.lastActivity, // Keep as string, Chat interface expects string
        title: chat.title || 'ungli-untitled'
      })) || [];
      return chats;
    } catch (error) {
      console.error('Error fetching chats:', error);
      // ⭐ No need to check for "TokenExpired" here as we're not using authService.fetchWithAuth
      return []; // Return empty array for any error
    }
  }

  async getChatMessages(chatId: string): Promise<ConversationTurn[]> {
    try {
      // ⭐ MODIFIED: Use plain fetch for bypassed endpoint
      const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}/messages`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || 'Failed to fetch messages');
      }

      const result: TransformedChatMessagesResponse = await response.json();

      if (result && result.messages) {
          return result.messages;
      } else {
          console.warn("Backend did not return 'messages' array in transformed response:", result);
          return [];
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
      // ⭐ No need to check for "TokenExpired" here
      return [];
    }
  }

  async sendMessage(data: SendMessageRequest): Promise<SendMessageSuccessResponse> {
    try {
      // ⭐ MODIFIED: Use plain fetch for bypassed endpoint
      const response = await fetch(`${API_BASE_URL}/api/chats/${data.chatId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: data.content,
        }),
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || 'Failed to send message');
      }

      const result: SendMessageSuccessResponse = await response.json();
      // ⭐ UPDATED: Check for 'result.message' instead of 'result.bot_message_instance'
      if (result && result.message) {
          return result;
      } else {
          console.error('Backend did not return expected message field:', result); // ⭐ Updated log
          // Throw a more specific error for this case
          throw new Error("Invalid response: message field missing."); // ⭐ Updated error message
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // ⭐ No need to check for "TokenExpired" here
      throw error; // Re-throw other errors for Chat.tsx to handle
    }
  }

  /**
   * Creates a new chat.
   * @param title Optional title for the new chat. Pass `undefined` or `null` if no title.
   */
  async createChat(title?: string | null): Promise<Chat | null> {
    try {
      const requestBody: { type: string; title?: string } = {
        type: 'direct', // For 1-on-1 messaging
      };
      if (typeof title === 'string' && title.trim() !== '') {
        requestBody.title = title.trim();
      }

      // ⭐ MODIFIED: Use plain fetch for bypassed endpoint
      const response = await fetch(`${API_BASE_URL}/api/chats`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || 'Failed to create chat');
      }

      const result = await response.json();
      if (result && result.chats && result.chats.length > 0) {
        const newChatData = result.chats[0]; // Access the first element of the 'chats' array
        return {
          id: newChatData._id, // Map backend `_id` to frontend `id`
          participants: newChatData.participants,
          lastMessage: newChatData.lastMessage,
          lastActivity: newChatData.lastActivity, // Keep as string
          title: newChatData.title || 'ungli-untitled'
        };
      } else {
        console.error("Backend did not return a valid 'chats' array with a new chat object in the response:", result);
        return null;
      }
    } catch (error) {
      console.error('Error creating chat:', error);
      // ⭐ No need to check for "TokenExpired" here
      return null; // Return null for other errors
    }
  }

  // Keep these as they were, they're already unauthenticated or don't rely on authService.fetchWithAuth
  async initChatSession(): Promise<any> {
      const response = await fetch(`${API_BASE_URL}/api/chat/init`);
      if (!response.ok) {
          const errorBody = await response.json();
          throw new Error(errorBody.detail || `Failed to initialize chat: ${response.statusText}`);
      }
      return response.json();
  }

  async triggerSearchPipeline(): Promise<any> {
      const response = await fetch(`${API_BASE_URL}/api/trigger_search_pipeline`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          // body: JSON.stringify({}) // Assuming no body is needed for this endpoint
      });
      if (!response.ok) {
          const errorBody = await response.json();
          throw new Error(errorBody.detail || `Failed to trigger search pipeline: ${response.statusText}`);
      }
      return response.json();
  }
}

export const chatService = new ChatService();