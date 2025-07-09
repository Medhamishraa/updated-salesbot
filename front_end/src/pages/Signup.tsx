import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { MessageCircle } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "@/hooks/use-toast";
import { authService } from "@/services/authService";

// Import Google OAuth components and hooks
import { GoogleOAuthProvider, useGoogleLogin } from '@react-oauth/google';

// Import Font Awesome icons for Google and Facebook
import { FaGoogle } from 'react-icons/fa';
import { FaFacebookF } from 'react-icons/fa'; // Correct import for Facebook icon

const generateStrongPassword = (): string => {
  const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+";
  const length = 12;
  let password = "";
  for (let i = 0; i < length; i++) {
    const randomIndex = Math.floor(Math.random() * charset.length);
    password += charset[randomIndex];
  }
  return password;
};

// Ensure these environment variables are correctly set in your .env.local or similar
const GOOGLE_CLIENT_ID_FRONTEND = import.meta.env.VITE_GOOGLE_CLIENT_ID_FRONTEND || 'YOUR_GOOGLE_CLIENT_ID_FROM_GOOGLE_CLOUD';
// ⭐ CHANGED from GITHUB_CLIENT_ID to FACEBOOK_APP_ID
const FACEBOOK_APP_ID = import.meta.env.VITE_FACEBOOK_APP_ID || 'YOUR_FACEBOOK_APP_ID';
const BACKEND_BASE_URL = import.meta.env.VITE_BACKEND_BASE_URL || 'http://127.0.0.1:8006'; // Make sure this matches your FastAPI backend URL

const SignupForm = () => {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    agreeToTerms: false
  });
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [suggestedPassword, setSuggestedPassword] = useState("");
  const navigate = useNavigate();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));

    // Generate suggested password only if email is valid and field is email
    if (name === "email" && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
      if (!suggestedPassword) { // Only generate if not already generated
        const generated = generateStrongPassword();
        setSuggestedPassword(generated);
      }
    } else if (name === "email" && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
      setSuggestedPassword(""); // Clear suggestion if email becomes invalid
    }
  };

  const handleUseSuggestedPassword = () => {
    setFormData(prev => ({ ...prev, password: suggestedPassword, confirmPassword: suggestedPassword })); // Also fill confirm
    setSuggestedPassword("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      toast({
        title: "Password mismatch",
        description: "Please make sure your passwords match.",
        variant: "destructive"
      });
      return;
    }

    if (!formData.agreeToTerms) {
      toast({
        title: "Terms required",
        description: "Please agree to the terms and conditions.",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);

    try {
      const result = await authService.signup({
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        password: formData.password
      });

      if (result.success) {
        toast({
          title: "Account created successfully!",
          description: "Welcome to UNGLI. You can now log in.",
        });
        navigate("/login"); // Redirect to login after successful signup
      } else {
        toast({
          title: "Signup failed",
          description: result.message || "An error occurred during signup.",
          variant: "destructive"
        });
      }
    } catch (error: any) {
      console.error("Signup error:", error);
      toast({
        title: "Error",
        description: error.message || "An unexpected error occurred. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Google Login Hook (remains the same)
  const googleLogin = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      console.log("Google login success:", tokenResponse);
      setIsLoading(true);
      try {
        const result = await authService.googleAuth(tokenResponse.access_token);
        if (result.success) {
          toast({
            title: "Login Successful",
            description: result.message || `Welcome, ${result.user?.firstName}!`,
          });
          navigate("/chat", { replace: true });
        } else {
          toast({
            title: "Google Login Failed",
            description: result.message || "An error occurred during Google login.",
            variant: "destructive",
          });
        }
      } catch (error: any) {
        console.error("Error during Google auth service call:", error);
        toast({
          title: "Google Login Error",
          description: error.message || "An unexpected error occurred during Google login. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    },
    onError: (errorResponse) => {
      console.error("Google login error:", errorResponse);
      toast({
        title: "Google Login Error",
        description: errorResponse.error_description || "Could not complete Google login.",
        variant: "destructive",
      });
    },
    flow: 'implicit',
  });

  // ⭐ Facebook Login Handler (replaces GitHub)
  const handleFacebookLogin = () => {
    // For a simple redirect, you can use the Facebook OAuth dialog URL.
    // Ensure your Facebook App settings have the correct redirect URI configured.
    // The `scope` parameter determines the permissions you're requesting.
    const facebookAuthUrl = `https://www.facebook.com/v19.0/dialog/oauth?` +
      `client_id=${FACEBOOK_APP_ID}` +
      `&redirect_uri=${BACKEND_BASE_URL}/api/auth/facebook/callback` + // Your backend callback URL
      `&scope=email,public_profile` + // Requesting email and public profile
      `&response_type=code`; // Use 'code' for server-side flow, 'token' for client-side implicit flow

    window.location.href = facebookAuthUrl;
    // Note: For a more robust client-side Facebook integration, you'd typically
    // use Facebook's JavaScript SDK, which handles pop-ups and token retrieval more smoothly.
    // This redirect approach relies on your backend to complete the OAuth flow.
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-center space-x-2 mb-8">
          <MessageCircle className="h-8 w-8 text-blue-600" />
          <span className="text-2xl font-bold text-gray-900">UNGLI</span>
        </div>

        <Card className="border-0 shadow-xl">
          <CardHeader className="space-y-1 text-center">
            <CardTitle className="text-2xl font-bold">Create your account</CardTitle>
            <CardDescription>Join thousands of users already using UNGLI</CardDescription>
          </CardHeader>
          <CardContent>
            {/* Social Login Buttons */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <Button
                variant="outline"
                className="w-full h-11 flex items-center justify-center space-x-2"
                onClick={() => googleLogin()}
                disabled={isLoading}
              >
                <FaGoogle className="h-5 w-5" />
                <span>Google</span>
              </Button>
              {/*  Facebook Button */}
              <Button
                variant="outline"
                className="w-full h-11 flex items-center justify-center space-x-2"
                onClick={handleFacebookLogin} // Trigger Facebook login
                disabled={isLoading}
              >
                <FaFacebookF className="h-5 w-5" /> {/* Facebook icon */}
                <span>Facebook</span>
              </Button>
            </div>

            {/* OR Divider */}
            <div className="relative mb-6">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  Or continue with email
                </span>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First name</Label>
                  <Input id="firstName" name="firstName" type="text" value={formData.firstName} onChange={handleInputChange} required className="h-11" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last name</Label>
                  <Input id="lastName" name="lastName" type="text" value={formData.lastName} onChange={handleInputChange} required className="h-11" />
                </div>
              </div>

              {suggestedPassword && (
                <div className="bg-gray-100 p-3 rounded-md text-sm text-gray-700 mb-2 flex items-center justify-between">
                  <p>Suggested Password: <span className="font-mono break-all">{suggestedPassword}</span></p>
                  <Button type="button" variant="ghost" size="sm" onClick={handleUseSuggestedPassword} className="ml-2">Use this</Button>
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" name="email" type="email" value={formData.email} onChange={handleInputChange} required className="h-11" />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Input id="password" name="password" type={showPassword ? "text" : "password"} value={formData.password} onChange={handleInputChange} required className="h-11 pr-20" />
                  {formData.password && (
                    <Button type="button" variant="ghost" className="absolute top-1 right-1 text-xs" onClick={() => setShowPassword(prev => !prev)}>
                      {showPassword ? "Hide" : "Show"}
                    </Button>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm password</Label>
                <Input id="confirmPassword" name="confirmPassword" type="password" value={formData.confirmPassword} onChange={handleInputChange} required className="h-11" />
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox id="terms" checked={formData.agreeToTerms} onCheckedChange={(checked) => setFormData(prev => ({ ...prev, agreeToTerms: checked as boolean }))} />
                <Label htmlFor="terms" className="text-sm text-gray-600">
                  I agree to the <Link to="/terms" className="text-blue-600 hover:underline">Terms of Service</Link> and <Link to="/privacy" className="text-blue-600 hover:underline">Privacy Policy</Link>
                </Label>
              </div>

              <Button type="submit" className="w-full h-11" disabled={isLoading}>
                {isLoading ? "Creating account..." : "Create account"}
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="mt-8 text-center">
          <Link to="/" className="text-sm text-gray-600 hover:text-gray-800 hover:underline">
            ← Back to home
          </Link>
        </div>
      </div>
    </div>
  );
};

const Signup = () => {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID_FRONTEND}>
      <SignupForm />
    </GoogleOAuthProvider>
  );
};

export default Signup;