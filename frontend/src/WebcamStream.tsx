import Webcam from "react-webcam";
import { useRef, useEffect, useState } from "react";

export default function WebcamStream() {
  const webcamRef = useRef<Webcam>(null);
  const [text, setText] = useState("");

  // capture every 500ms
  useEffect(() => {
    const interval = setInterval(() => {
      captureAndSend();
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const captureAndSend = async () => {
    if (!webcamRef.current) return;

    // take screenshot (base64)
    const imageSrc = webcamRef.current.getScreenshot();

    if (!imageSrc) return;

    const res = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        image: imageSrc,
      }),
    });

    const result = await res.json();
    setText(result.text);
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <h2 className="text-2xl font-bold">American Sign Language</h2>

      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="w-200"
      />

      <div className="text-xl font-mono border p-4 w-200 text-center">
        {text || "Detecting..."}
      </div>
    </div>
  );
}