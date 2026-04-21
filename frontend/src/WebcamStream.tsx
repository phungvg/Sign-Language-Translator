import Webcam from "react-webcam";
import { useRef, useEffect, useState } from "react";

export default function WebcamStream() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [label, setLabel] = useState("");
  const [sentence, setSentence] = useState("");
  const [currentWord, setCurrentWord] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);

  const [confidence, setConfidence] = useState<string>("");
  const [hand, setHand] = useState<string>("");
  const [type, setType] = useState<string>("");
  const currentWordRef = useRef<string>("");

  // capture every 300ms
  useEffect(() => {
    const interval = setInterval(() => {
      captureAndSend();
    }, 300);

    return () => clearInterval(interval);
  }, []);

  // draw landmarks
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;

    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length < 21) return;

    // draw points
    landmarks.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * width, y * height, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    });

    // skeleton
    const connections = [
      [0,1],[1,2],[2,3],[3,4],
      [5,6],[6,7],[7,8],
      [9,10],[10,11],[11,12],
      [13,14],[14,15],[15,16],
      [17,18],[18,19],[19,20],
      [0,5],[5,9],[9,13],[13,17],[17,0]
    ];

    connections.forEach(([a, b]) => {
      const [x1, y1] = landmarks[a];
      const [x2, y2] = landmarks[b];

      ctx.beginPath();
      ctx.moveTo(x1 * width, y1 * height);
      ctx.lineTo(x2 * width, y2 * height);
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.stroke();
    });

  }, [landmarks]);

  const captureAndSend = async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    // create image
    const img = new Image();
    img.src = imageSrc;

    img.onload = async () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      canvas.width = img.width;
      canvas.height = img.height;

      if (!ctx) return;

      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);

      ctx.drawImage(img, 0, 0);

      const flippedImage = canvas.toDataURL("image/jpeg");

      await sendToBackend(flippedImage);
    };
  }

  const correctAndFlush = async () => {
    const word = currentWordRef.current;
    if (!word) return;

    const res = await fetch("http://127.0.0.1:8000/correct", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ word }),
    });
    const data = await res.json();
    const corrected: string = data.corrected || word;

    setSentence(s => s + corrected + " ");
    currentWordRef.current = "";
    setCurrentWord("");
    setSuggestions([]);
  };

  const sendToBackend = async (imageSrc: string) => {
    const ui_res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ image: imageSrc }),
    });

    const result = await ui_res.json();

    if (!result) {
      setLandmarks(null);
      setLabel("");
      setHand("");
      setType("");
      setConfidence("");
      return;
    }

    if (result.landmarks) {
      const pairs = [];
      for (let i = 0; i < result.landmarks.length; i += 2) {
        pairs.push([result.landmarks[i], result.landmarks[i + 1]]);
      }
      setLandmarks(pairs);
    } else if (result.type === "space" && !result.hand) {
      // special case for space gesture which may not have landmarks
      setLandmarks(null);
      setHand("")
    }
    else {
      setLandmarks(null);
      setLabel("");
      return;
    }

    if (!result.label) {
      return;
    }

    const label = result.label;
    setLabel(label);
    setConfidence(result.confidence);
    if (result.hand)
      setHand(result.hand);
    setType(result.type);

    if (result.confidence < 0.5) {
      return; // ignore low confidence predictions
    }

    if (label === "space") {
      await correctAndFlush();
    } else if (label === "del") {
      const updated = currentWordRef.current.slice(0, -1);
      currentWordRef.current = updated;
      setCurrentWord(updated);
    } else {
      const updated = currentWordRef.current + label;
      currentWordRef.current = updated;
      setCurrentWord(updated);

      // fetch live suggestions for the partial word
      fetch("http://127.0.0.1:8000/suggest", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ prefix: updated }),
      })
        .then(r => r.json())
        .then(data => setSuggestions(data.suggestions || []));
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <h2 className="text-2xl font-bold">American Sign Language</h2>

      <div className="relative w-200">
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 400, height: 300 }}
          // mirror the video feed
          className="w-full scale-x-[-1]"
        />

        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* current letter prediction */}
      <div className="text-xl font-mono border p-4 text-center">
        {label || "Detecting..."}
      </div>

      {/* current word being signed */}
      <div className="text-lg font-mono text-blue-600">
        {currentWord ? `Signing: ${currentWord}` : ""}
      </div>

      <div className="text-sm text-gray-600">
        (Click suggestion to auto-complete)
      </div>
      {/* live word suggestions */}
      {suggestions.length > 0 && (
        <div className="flex gap-2 flex-wrap justify-center">
          {suggestions.map(s => (
            <button
              key={s}
              className="px-3 py-1 border rounded text-sm font-mono hover:bg-gray-100"
              onClick={() => {
                setSentence(prev => prev + s + " ");
                currentWordRef.current = "";
                setCurrentWord("");
                setSuggestions([]);
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* sentence */}
      <div className="text-xl font-mono border p-4 w-200 text-center wrap-break-word overflow-y-auto whitespace-pre-wrap">
        {sentence || "No signs detected yet."}
      </div>

      <div className="text-sm text-gray-600">
        Confidence: {confidence} | Hand: {hand} | Type: {type}
      </div>
      
    </div>
    
  );
}