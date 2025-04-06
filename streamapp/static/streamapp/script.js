console.log("✅ Script loaded");

// DOM references
const video = document.getElementById('video-stream');
const predictionText = document.getElementById("prediction");

// Initialize WebSocket connection
const ws = new WebSocket(`ws://${window.location.host}/ws/stream/`);

ws.onopen = () => {
    console.log("🔌 WebSocket connected");
};

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        console.log("📩 Received from server:", data);
        
        if (data.prediction) {
            predictionText.textContent = data.prediction;
            
            // Optional: add visual feedback for detected chords
            if (data.prediction !== "No chord detected" && 
                data.prediction !== "No hands detected" &&
                data.prediction !== "Waiting for hand gestures...") {
                // Flash the prediction to indicate successful detection
                predictionText.classList.add('detected');
                setTimeout(() => {
                    predictionText.classList.remove('detected');
                }, 1000);
            }
        } else if (data.error) {
            console.error("❌ Server error:", data.error);
            predictionText.textContent = "Error: " + data.error;
        }
    } catch (err) {
        console.error("❌ Failed to parse server message:", err);
    }
};

ws.onerror = (err) => {
    console.error("❌ WebSocket error:", err);
    predictionText.textContent = "WebSocket error occurred";
};

ws.onclose = () => {
    console.warn("⚠️ WebSocket connection closed");
    predictionText.textContent = "Connection closed. Refresh to reconnect.";
};

// Request camera access and start streaming
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
        console.log("📷 Camera access granted");
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            startFrameSending(); // begin sending frames
        };
    })
    .catch((err) => {
        console.error("❌ Camera access denied:", err);
        predictionText.textContent = "Camera permission is required";
        alert("Camera permission is required for gesture recognition.");
    });

// Function to send frames to the backend periodically
function startFrameSending() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    setInterval(() => {
        if (video.readyState !== video.HAVE_ENOUGH_DATA || ws.readyState !== WebSocket.OPEN) {
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64Image = canvas.toDataURL('image/jpeg', 0.8); // Lower quality for better performance

        ws.send(JSON.stringify({ frame: base64Image }));
    }, 200); // Send every 200ms for better performance
}
