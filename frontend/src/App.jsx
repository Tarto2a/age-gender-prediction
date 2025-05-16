import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [uploadedImageURL, setUploadedImageURL] = useState(null);

  // Start webcam
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    });
  }, []);

  // Capture and send frame every 2 seconds
  useEffect(() => {
    const interval = setInterval(captureFrame, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    // Display image on screen
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedImageURL(reader.result);
    };
    reader.readAsDataURL(file);

    try {
      const res = await axios.post("http://localhost:8000/predict/", formData);
      if (res.data.error) {
        setError(res.data.error);
        setPrediction(null);
      } else {
        setPrediction(res.data);
        setError("");
      }
    } catch (err) {
      setError(`Prediction failed. Please try again.${err}`);
      setPrediction(null);
    }
  };

  const captureFrame = async () => {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const res = await axios.post(
          "http://localhost:8000/predict/",
          formData
        );
        if (res.data.error) {
          setError(res.data.error);
          setPrediction(null);
        } else {
          setPrediction(res.data);
          setError("");
        }
      } catch (err) {
        console.error("Prediction error:", err);
        setError("Prediction failed.");
        setPrediction(null);
      }
    }, "image/jpeg");
  };

  // Draw overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    const drawOverlay = () => {
      const video = videoRef.current;
      const ctx = canvas.getContext("2d");
    
      if (!video || !canvas) return;
    
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    
      // Mirror the context (like video)
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    
      if (prediction?.box) {
        const { x, y, w, h } = prediction.box;
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
    
        // Unmirror only the text
        ctx.save();
        ctx.translate(canvas.width, 0); // Move back to original
        ctx.scale(-1, 1); // Unflip for text
        ctx.fillStyle = "lime";
        ctx.font = "16px Arial";
        ctx.fillText(
          `${prediction.gender}, ${prediction.age_range}`,
          canvas.width - x,
          y - 10
        );
        ctx.restore();
      }
    
      ctx.restore(); // Restore original context
    
      requestAnimationFrame(drawOverlay);
    };    

    drawOverlay();
  }, [prediction]);

  return (
    <div className="app-container">
      <h1>age & gender prediction</h1>
      <div className="upload-container">
        <label className="upload-label">
          Upload Image:
          <input type="file" accept="image/*" onChange={handleImageUpload} />
        </label>
        
      </div>

      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="video-element"
        />
        <canvas ref={canvasRef} className="canvas-element" />
      </div>

      <div className="prediction-display">
        {error ? (
          <span style={{ color: "red" }}>{error}</span>
        ) : prediction ? (
          `${prediction.gender}, ${prediction.age_range}`
        ) : (
          "Waiting for prediction..."
        )}
      </div>
      {uploadedImageURL && (
          <div className="uploaded-image-container">
            <img
              src={uploadedImageURL}
              alt="Uploaded"
              className="uploaded-image"
              style={{
                maxWidth: "100%",
                border: "1px solid #ccc",
                marginTop: "10px",
              }}
            />
          </div>
        )}
    </div>
  );
}

export default App;
