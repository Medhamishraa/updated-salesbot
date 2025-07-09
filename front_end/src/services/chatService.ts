// /Users/atishaymalik/Desktop/ungli/UNGLI/front_end/src/services/chatService.ts

import { authService } from './authService'; // ⭐ Import authService

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
  // ⭐ Use authService's centralized fetchWithAuth method
  // This binds the method to authService's instance to ensure 'this' context is correct
  private fetchWithAuth = authService.fetchWithAuth.bind(authService);

  async getUserChats(): Promise<Chat[]> {
    try {
      // ⭐ Use fetchWithAuth directly. It handles auth headers and 401 redirection.
      const response = await this.fetchWithAuth(`${API_BASE_URL}/api/chats`, {
        method: 'GET',
      });

      // No need for handleUnauthorizedResponse here, fetchWithAuth already called it.
      // If a 401 occurred, fetchWithAuth would have thrown "TokenExpired" and redirected.

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
      // ⭐ Re-throw only if it's the "TokenExpired" error, otherwise let Chat.tsx handle general API errors
      if (error instanceof Error && error.message === "TokenExpired") {
        throw error; // This will trigger the Chat.tsx top-level auth guard/redirect if not already handled
      }
      return []; // Return empty array for non-auth errors
    }
  }

  async getChatMessages(chatId: string): Promise<ConversationTurn[]> {
    try {
      // ⭐ Use fetchWithAuth directly
      const response = await this.fetchWithAuth(`${API_BASE_URL}/api/chats/${chatId}/messages`, {
        method: 'GET',
      });

      const result: TransformedChatMessagesResponse = await response.json();

      if (result && result.messages) {
          return result.messages;
      } else {
          console.warn("Backend did not return 'messages' array in transformed response:", result);
          return [];
      }
    } catch (error) {
      console.error('Error fetching messages:', error);
      // ⭐ Re-throw only if it's the "TokenExpired" error
      if (error instanceof Error && error.message === "TokenExpired") {
        throw error;
      }
      return [];
    }
  }

  async sendMessage(data: SendMessageRequest): Promise<SendMessageSuccessResponse> {
    try {
      // ⭐ Use fetchWithAuth directly
      const response = await this.fetchWithAuth(`${API_BASE_URL}/api/chats/${data.chatId}/messages`, {
        method: 'POST',
        body: JSON.stringify({
          content: data.content,
        }),
      });

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
      // ⭐ Re-throw only if it's the "TokenExpired" error
      if (error instanceof Error && error.message === "TokenExpired") {
        throw error;
      }
      throw error; // Re-throw other errors for Chat.tsx to handle
    }
  }

  /**
   * Creates a new chat.
   * @param title Optional title for the new chat. Pass `undefined` or `null` if no title.
   */
  async createChat(title?: string | null): Promise<Chat | null> {
    try {
      // No need for the manual token check here; authService.fetchWithAuth handles it.
      const requestBody: { type: string; title?: string } = {
        type: 'direct', // For 1-on-1 messaging
      };
      if (typeof title === 'string' && title.trim() !== '') {
        requestBody.title = title.trim();
      }

      // ⭐ Use fetchWithAuth directly
      const response = await this.fetchWithAuth(`${API_BASE_URL}/api/chats`, {
        method: 'POST',
        body: JSON.stringify(requestBody),
      });

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
      // ⭐ Re-throw only if it's the "TokenExpired" error
      if (error instanceof Error && error.message === "TokenExpired") {
        throw error;
      }
      return null; // Return null for other errors
    }
  }
}

export const chatService = new ChatService();