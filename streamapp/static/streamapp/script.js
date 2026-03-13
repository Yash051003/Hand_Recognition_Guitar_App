console.log("Script loaded");

const video = document.getElementById('video-stream');
const predictionText = document.getElementById("prediction");

// Audio state and chord map are defined in index.html inline script
// window.audioEnabled, window.chordElementMap, unlockAudio(), playChordSound()
// are all available globally

// Flip video display (cosmetic only)
video.style.transform = "scaleX(-1)";
predictionText.textContent = "Click Enable Audio, then make a gesture";

// WebSocket
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream/`);

ws.onopen = () => console.log("WebSocket connected");

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        
        // ADD THESE 3 DEBUG LINES:
        console.log("Raw prediction:", JSON.stringify(data.prediction));
        console.log("audioEnabled:", window.audioEnabled);
        console.log("Is chord?", data.prediction && data.prediction !== "No chord detected" && data.prediction !== "No hands detected");

        if (data.prediction) {
            predictionText.textContent = data.prediction;
            const isChord = !["No chord detected", "No hands detected",
                              "Waiting for hand gestures..."].includes(data.prediction);
            if (isChord) {
                predictionText.classList.add('detected');
                setTimeout(() => predictionText.classList.remove('detected'), 1000);
                playChordSound(data.prediction);
            } else {
                lastChord = null;
            }
        }
    } catch (e) {
        console.error("Parse error:", e);
    }
};

ws.onerror = () => { predictionText.textContent = "WebSocket error"; };
ws.onclose = () => { predictionText.textContent = "Connection closed. Refresh to reconnect."; };

// Camera
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = () => { video.play(); startFrameSending(); };
    })
    .catch(() => { predictionText.textContent = "Camera permission required"; });

// Send raw (unflipped) frames - backend swaps Left/Right labels
function startFrameSending() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    setInterval(() => {
        if (video.readyState !== video.HAVE_ENOUGH_DATA || ws.readyState !== WebSocket.OPEN) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ws.send(JSON.stringify({ frame: canvas.toDataURL('image/jpeg', 0.8) }));
    }, 300);
}