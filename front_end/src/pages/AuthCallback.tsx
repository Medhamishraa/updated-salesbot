// src/pages/AuthCallback.tsx
import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { toast } from "@/hooks/use-toast";

const AuthCallback = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get('token');
    const firstName = params.get('firstName');
    const lastName = params.get('lastName');
    const email = params.get('email');
    const message = params.get('message');
    const error = params.get('error');

    if (error) {
      toast({
        title: "Authentication Error",
        description: decodeURIComponent(error.replace(/_/g, ' ')), // Decode error message
        variant: "destructive"
      });
      navigate('/login', { replace: true }); // Go back to login if there's an error
      return;
    }

    if (token) {
      // Assuming your authService has a way to store the token and user info
      // For example, localStorage or a global state management (Context API, Redux, Zustand)
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify({ firstName, lastName, email }));

      toast({
        title: "Login Successful",
        description: message || `Welcome, ${firstName}!`,
      });
      navigate('/chat', { replace: true }); // Redirect to chat after successful login
    } else {
      toast({
        title: "Authentication Failed",
        description: "No token received. Please try again.",
        variant: "destructive"
      });
      navigate('/login', { replace: true }); // Fallback to login
    }
  }, [location, navigate]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <p>Processing Google login...</p>
    </div>
  );
};

export default AuthCallback;