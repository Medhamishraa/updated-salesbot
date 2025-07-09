import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, MessageCircle, User, PlayCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { useKeenSlider } from "keen-slider/react";
import "keen-slider/keen-slider.min.css";
import { useEffect } from "react";

// TODO: Replace these placeholder paths with your actual images for each slide.
const showcaseImages = [
  "/image1.jpg", // Corrected path
  "/image2.jpg"  // Corrected path
];

const Landing = () => {
  const [sliderRef, instanceRef] = useKeenSlider<HTMLDivElement>({
    loop: true,
    mode: "free-snap",
    slides: {
      perView: 1.25,
      spacing: 16,
    },
    breakpoints: {
      "(min-width: 768px)": {
        slides: {
          perView: 2.5,
          spacing: 24,
        },
      },
      "(min-width: 1024px)": {
        slides: {
          perView: 3.5,
          spacing: 32,
        },
      },
    },
  });

  // Autoplay functionality
  useEffect(() => {
    const interval = setInterval(() => {
      instanceRef.current?.next();
    }, 3000);
    return () => clearInterval(interval);
  }, [instanceRef]);

  return (
    // Main container with gradient and new dotted grid background
    <div
      className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50"
      style={{
        backgroundImage: 'radial-gradient(circle at 1px 1px, #dbeafe 1px, transparent 0)',
        backgroundSize: '2rem 2rem'
      }}
    >
      {/* Navigation */}
      <nav className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <MessageCircle className="h-8 w-8 text-blue-600" />
            <Link to="/" className="text-2xl font-bold text-gray-900">UNGLI</Link>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/login"><Button variant="outline">Login</Button></Link>
            <Link to="/signup"><Button>Sign Up</Button></Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
            Freshly Curated leads
            <span className="text-blue-600 block">through extreme deep research</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Ungli talks to you to understand your business and product/service in depth,
            then goes to every corner of the internet to find you easy to convert and high ticket size leads.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/signup">
              <Button size="lg" className="text-lg px-8 py-4">
                Start AI Consultation <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Button size="lg" variant="outline" className="text-lg px-8 py-4 gap-2">
              <PlayCircle className="w-6 h-6" />
              Watch Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Slider Section */}
      <section className="py-16">
        <div ref={sliderRef} className="keen-slider">
          {showcaseImages.map((src, index) => (
            <div key={index} className="keen-slider__slide">
              <div className="bg-white/40 backdrop-blur-lg rounded-3xl border border-white/50 shadow-xl p-2 h-[280px] md:h-[320px] lg:h-[350px]">
                <img
                  src={src}
                  // Corrected line: Use standard backticks (`) for the template literal
                  alt={`Showcase slide ${index + 1}`}
                  className="rounded-2xl w-full h-full object-cover"
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Why Choose UNGLI?</h2>
          <p className="text-xl text-gray-600">
            Simplest, yet the most powerful lead generation, enrichment and qualification engine
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          {[{
            icon: <MessageCircle className="h-8 w-8 text-blue-600" />,
            title: "Understands your Business first",
            desc: "Before we offer strategies or solutions, we take the time to understand your business  from the inside out. Your success depends on it — and so does ours."
          },{
            icon: <User className="h-8 w-8 text-green-600" />,
            title: "Generate Leads-Hot & Fresh",
            desc: "Freshly captured. Highly targeted. We help you connect with leads that are warm and ready to talk business."
          }, {
            icon: <ArrowRight className="h-8 w-8 text-purple-600" />,
            title: "Deep & Expensive Research & Reasoning",
            desc: "We conduct intensive research and critical reasoning that demand time, expertise, and precision — ensuring every solution is grounded in evidence, not assumptions."
          }].map((item, i) => (
            <Card key={i} className="border-0 shadow-lg hover:shadow-xl transition-shadow duration-300">
              <CardHeader className="text-center">
                <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
                     style={{ backgroundColor: i === 0 ? "#DBEAFE" : i === 1 ? "#D1FAE5" : "#EDE9FE" }}>
                  {item.icon}
                </div>
                <CardTitle className="text-xl">{item.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center text-base">
                  {item.desc}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
          {/* Infographic Image */}
        <div className="mt-12 flex justify-center">
          <img
            src="/infograph.jpg"
            alt="UNGLI process infographic"
            className="rounded-xl shadow-lg w-full max-w-3xl"
          />
        </div>
      </div>
      </section>

      {/* CTA */}
      <section className="bg-blue-600 text-white py-20">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-4">
            Ready to accelerate your market penetration?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Join thousands of users who are already experiencing the future of AI chat.
          </p>
          <Link to="/signup">
            <Button size="lg" variant="secondary" className="text-lg px-8 py-4">
              Get Started <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-center space-x-2 mb-8">
            <MessageCircle className="h-6 w-6" />
            <Link to="/" className="text-xl font-bold">UNGLI</Link>
          </div>
          <div className="text-center text-gray-400">
            <p>&copy; 2024 Ungli. All rights reserved.</p>
          </div>
        </div>
      </footer>

      {/* Chat Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <Link to="/chat">
          <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg px-6 py-4 rounded-full flex items-center space-x-2">
            <MessageCircle className="w-5 h-5" />
            <span>Chat Now</span>
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default Landing;