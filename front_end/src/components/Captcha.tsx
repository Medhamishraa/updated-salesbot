import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RefreshCw } from "lucide-react";

interface CaptchaProps {
  onVerify: (token: string) => void;
  onError?: () => void;
}

const Captcha = ({ onVerify, onError }: CaptchaProps) => {
  const [captchaText, setCaptchaText] = useState("");
  const [userInput, setUserInput] = useState("");
  const [isVerified, setIsVerified] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const generateCaptcha = () => {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let result = "";
    for (let i = 0; i < 6; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    setCaptchaText(result);
    setUserInput("");
    setIsVerified(false);
    return result;
  };

  const drawCaptcha = (text: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add noise lines
    for (let i = 0; i < 5; i++) {
      ctx.strokeStyle = `hsl(${Math.random() * 360}, 50%, 70%)`;
      ctx.beginPath();
      ctx.moveTo(Math.random() * canvas.width, Math.random() * canvas.height);
      ctx.lineTo(Math.random() * canvas.width, Math.random() * canvas.height);
      ctx.stroke();
    }

    // Draw text
    ctx.font = "24px Arial";
    ctx.fillStyle = "#333";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    
    // Add some distortion to each character
    const chars = text.split("");
    const spacing = canvas.width / (chars.length + 1);
    
    chars.forEach((char, index) => {
      const x = spacing * (index + 1) + (Math.random() - 0.5) * 10;
      const y = canvas.height / 2 + (Math.random() - 0.5) * 10;
      const rotation = (Math.random() - 0.5) * 0.3;
      
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotation);
      ctx.fillText(char, 0, 0);
      ctx.restore();
    });
  };

  const handleVerify = () => {
    if (userInput.toLowerCase() === captchaText.toLowerCase()) {
      setIsVerified(true);
      onVerify(captchaText);
    } else {
      onError?.();
      generateCaptcha();
    }
  };

  const handleRefresh = () => {
    const newCaptcha = generateCaptcha();
    drawCaptcha(newCaptcha);
  };

  useEffect(() => {
    const newCaptcha = generateCaptcha();
    drawCaptcha(newCaptcha);
  }, []);

  useEffect(() => {
    if (captchaText) {
      drawCaptcha(captchaText);
    }
  }, [captchaText]);

  if (isVerified) {
    return (
      <div className="flex items-center space-x-2 text-green-600">
        <span className="text-sm">✓ CAPTCHA verified</span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center space-x-2">
        <canvas
          ref={canvasRef}
          width={200}
          height={60}
          className="border border-gray-300 rounded bg-gray-50"
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleRefresh}
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex space-x-2">
        <Input
          type="text"
          placeholder="Enter CAPTCHA"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          className="flex-1"
        />
        <Button
          type="button"
          onClick={handleVerify}
          disabled={!userInput.trim()}
        >
          Verify
        </Button>
      </div>
    </div>
  );
};

export default Captcha;