import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { MessageCircle } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "@/hooks/use-toast";
import { authService } from "@/services/authService.ts";
import Captcha from "@/components/Captcha";

import { GoogleLogin, CredentialResponse } from '@react-oauth/google';

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [captchaVerified, setCaptchaVerified] = useState(false);
  const [captchaToken, setCaptchaToken] = useState("");
  const navigate = useNavigate();

  const handleCaptchaVerify = (token: string) => {
    setCaptchaVerified(true);
    setCaptchaToken(token);
  };

  const handleCaptchaError = () => {
    setCaptchaVerified(false);
    setCaptchaToken("");
    toast({
      title: "CAPTCHA verification failed",
      description: "Please try again.",
      variant: "destructive"
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!captchaVerified) {
      toast({
        title: "CAPTCHA required",
        description: "Please complete the CAPTCHA verification.",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);

    try {
      const result = await authService.login({
        email,
        password,
        captchaToken
      });

      if (result.success) {
        toast({
          title: "Welcome back!",
          description: "You have successfully logged in.",
        });
        navigate("/chat");
      } else {
        toast({
          title: "Login failed",
          description: result.message || "Invalid credentials.",
          variant: "destructive"
        });
        // Reset CAPTCHA on failed login
        setCaptchaVerified(false);
        setCaptchaToken("");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse: CredentialResponse) => {
    console.log("Google login success, credential response:", credentialResponse);
    if (credentialResponse.credential) {
      setIsLoading(true); // Start loading for backend call
      try {
        const authResult = await authService.googleAuth(credentialResponse.credential);
        if (authResult.success) {
          toast({
            title: "Welcome back!",
            description: authResult.message || "You have successfully logged in with Google.",
          });
          navigate("/chat");
        } else {
          toast({
            title: "Google Login Failed",
            description: authResult.message || "Something went wrong with Google login.",
            variant: "destructive"
          });
        }
      } catch (error) {
        console.error("Error calling authService.googleAuth:", error);
        toast({
          title: "Error",
          description: "An unexpected error occurred during Google login. Please try again.",
          variant: "destructive"
        });
      } finally {
        setIsLoading(false);
      }
    } else {
      console.error("Google credential missing from response.");
      toast({
        title: "Google Login Failed",
        description: "Could not get Google credentials. Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleGoogleError = () => {
    console.error("Google Login Failed (frontend side)");
    toast({
      title: "Google Login Failed",
      description: "Could not complete Google login. Please try again.",
      variant: "destructive"
    });
  };

  const handleFacebookLogin = async () => {
    toast({
      title: "Facebook Login",
      description: "Facebook OAuth integration would be implemented here.",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex items-center justify-center space-x-2 mb-8">
          <MessageCircle className="h-8 w-8 text-blue-600" />
          <span className="text-2xl font-bold text-gray-900">UNGLI</span>
        </div>

        <Card className="border-0 shadow-xl">
          <CardHeader className="space-y-1 text-center">
            <CardTitle className="text-2xl font-bold">Welcome back</CardTitle>
            <CardDescription>
              Enter your credentials to access your account
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="h-11"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="h-11"
                />
              </div>
              
              {/* CAPTCHA Component */}
              <div className="space-y-2">
                <Label>Security Verification</Label>
                <Captcha 
                  onVerify={handleCaptchaVerify}
                  onError={handleCaptchaError}
                />
              </div>

              <div className="flex items-center justify-between">
                <Link 
                  to="/forgot-password" 
                  className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
                >
                  Forgot password?
                </Link>
              </div>
              <Button
                type="submit"
                className="w-full h-11"
                disabled={isLoading || !captchaVerified}
              >
                {isLoading ? "Signing in..." : "Sign in"}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                Don't have an account?{" "}
                <Link 
                  to="/signup" 
                  className="text-blue-600 hover:text-blue-800 font-medium hover:underline"
                >
                  Sign up
                </Link>
              </p>
            </div>

            <div className="mt-6">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-white text-gray-500">Or continue with</span>
                </div>
              </div>
              
              <div className="mt-4 grid grid-cols-2 gap-3">
                {/* ⭐ CORRECTED: GoogleLogin Component without render prop */}
                <GoogleLogin
                  onSuccess={handleGoogleSuccess}
                  onError={handleGoogleError}
                  theme="outline"
                  size="large"
                  // width and text-align are not direct props, the library handles these internally for its button
                  // For width, the button will try to fit the container. You might need to adjust the grid or wrap in a div if exact width is crucial.
                  locale="en"
                  // The disabled prop is also managed internally by GoogleLogin for its own flow.
                />

                {/* Facebook Login Button (no change here, as it's still a placeholder) */}
                <Button 
                  variant="outline" 
                  className="h-11"
                  onClick={handleFacebookLogin}
                  disabled={isLoading}
                  type="button"
                >
                  <svg className="h-4 w-4 mr-2" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm3.208 8.167h-1.406c-.461 0-.693.303-.693.754v.982h2.106l-.332 2.158h-1.774v5.52h-2.16v-5.52h-1.423v-2.158h1.423v-.921c0-1.768.966-2.738 2.81-2.738h1.764v2.095z"/>
                  </svg>
                  Facebook
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="mt-8 text-center">
          <Link 
            to="/" 
            className="text-sm text-gray-600 hover:text-gray-800 hover:underline"
          >
            ← Back to home
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;