import Webcam from "react-webcam";
import { useRef, useEffect, useState } from "react";

export default function WebcamStream() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [text, setText] = useState("");
  const [sentence, setSentence] = useState("");
  const [landmarks, setLandmarks] = useState<number[][] | null>(null);

  const [confidence, setConfidence] = useState<string>("");
  const [hand, setHand] = useState<string>("");
  const [type, setType] = useState<string>("");

  const lastLabelRef = useRef<string>("");
  const lastTimeRef = useRef<number>(0);

  // capture every 500ms
  useEffect(() => {
    const interval = setInterval(() => {
      captureAndSend();
    }, 300);

    return () => clearInterval(interval);
  }, []);

  // draw landmarks
  useEffect(() => {
    if (!canvasRef.current || !landmarks || landmarks.length < 21) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;

    ctx.clearRect(0, 0, width, height);

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

  const sendToBackend = async (imageSrc: string) => {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        image: imageSrc,
      }),
    });

    const result = await res.json();

    const pred = result.prediction;

    if (!pred) return;

    const label = pred.label;
    const conf = pred.confidence;

    setText(label);
    setConfidence(conf);
    setHand(pred.hand);
    setType(pred.type);

    // convert flat [x,y,...] → [[x,y],...]
    if (pred.landmarks) {
      console.log("LANDMARKS FROM API:", pred.landmarks);
      const pairs = [];
      for (let i = 0; i < pred.landmarks.length; i += 2) {
        pairs.push([pred.landmarks[i], pred.landmarks[i + 1]]);
      }
      setLandmarks(pairs);
    }

    // TODO: Build sentence

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
        {text || "Detecting..."}
      </div>

      {/* sentence */}
      <div className="text-xl font-mono border p-4 w-200 text-center">
        {/* TODO: Display the sentence */}
        {"..."} 
      </div>

      <div className="text-sm text-gray-600">
        Confidence: {confidence} | Hand: {hand} | Type: {type}
      </div>
    </div>
  );
}