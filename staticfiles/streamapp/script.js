console.log("✅ Script loaded");

// DOM references
const video = document.getElementById('video-stream');
const predictionText = document.getElementById("prediction");
const enableAudioBtn = document.getElementById('enable-audio');

// Audio setup
let audioEnabled = false;
let audioContext = null;

// Chord to audio element mapping
const chordAudioMap = {
    // Major chords
    "A": "chord-A",
    "B": "chord-B", 
    "C": "chord-C",
    "D": "chord-D",
    "E": "chord-E",
    "F": "chord-F",
    "G": "chord-G",
    
    // Minor chords
    "Am": "chord-Am",
    "Bm": "chord-Bm",
    "Cm": "chord-Cm", 
    "Dm": "chord-Dm",
    "Em": "chord-Em",
    "Fm": "chord-Fm",
    "Gm": "chord-Gm",
    
    // Flat chords (convert symbols to safe IDs)
    "A♭": "chord-Ab",
    "B♭": "chord-Bb",
    "D♭": "chord-Db", 
    "E♭": "chord-Eb",
    "G♭": "chord-Gb",
    
    // Sharp chords (convert symbols to safe IDs)
    "A#": "chord-As",
    "C#": "chord-Cs",
    "D#": "chord-Ds",
    "F#": "chord-Fs", 
    "G#": "chord-Gs"
};

// Apply the horizontal flip to the video element
video.style.transform = "scaleX(-1)";

// Initialize welcome message
predictionText.textContent = "Make a chord gesture";

// Audio enable button functionality
enableAudioBtn.addEventListener('click', async () => {
    if (!audioEnabled) {
        try {
            // Create audio context (required for web audio)
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Test audio with a brief silent sound to "unlock" audio
            const buffer = audioContext.createBuffer(1, 1, 22050);
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.start();
            
            audioEnabled = true;
            enableAudioBtn.textContent = "🔊 Audio Enabled";
            enableAudioBtn.classList.add('enabled');
            
            console.log("✅ Audio enabled successfully");
        } catch (error) {
            console.error("❌ Failed to enable audio:", error);
            alert("Failed to enable audio. Please try again.");
        }
    }
});

// Chord playing state management
let currentlyPlayingAudio = null;
let lastPlayedChord = null;
let lastPlayTime = 0;
const CHORD_COOLDOWN = 1000; // 1 second cooldown between same chord

// Function to stop all currently playing audio
function stopAllAudio() {
    const allAudioElements = document.querySelectorAll('#audio-container audio');
    allAudioElements.forEach(audio => {
        if (!audio.paused) {
            audio.pause();
            audio.currentTime = 0;
        }
    });
    currentlyPlayingAudio = null;
}

// Function to play chord sound with debouncing
function playChordSound(chordName) {
    if (!audioEnabled) {
        console.log("🔇 Audio not enabled, skipping sound");
        return;
    }
    
    const currentTime = Date.now();
    
    // Prevent spam playing of the same chord
    if (lastPlayedChord === chordName && (currentTime - lastPlayTime) < CHORD_COOLDOWN) {
        console.log(`🔄 Chord ${chordName} in cooldown, skipping`);
        return;
    }
    
    const audioElementId = chordAudioMap[chordName];
    if (!audioElementId) {
        console.warn(`⚠️ No audio mapping found for chord: ${chordName}`);
        return;
    }
    
    const audioElement = document.getElementById(audioElementId);
    if (!audioElement) {
        console.warn(`⚠️ Audio element not found: ${audioElementId}`);
        return;
    }
    
    try {
        // Stop any currently playing audio first
        stopAllAudio();
        
        // Reset audio to beginning and play
        audioElement.currentTime = 0;
        audioElement.volume = 0.7; // Reduce volume slightly
        
        const playPromise = audioElement.play();
        
        if (playPromise !== undefined) {
            playPromise
                .then(() => {
                    console.log(`🎵 Playing chord: ${chordName}`);
                    currentlyPlayingAudio = audioElement;
                    lastPlayedChord = chordName;
                    lastPlayTime = currentTime;
                })
                .catch(error => {
                    console.error(`❌ Failed to play ${chordName}:`, error);
                });
        }
    } catch (error) {
        console.error(`❌ Error playing chord ${chordName}:`, error);
    }
}

// Initialize WebSocket connection
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream/`);

ws.onopen = () => {
    console.log("🔌 WebSocket connected");
};

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        console.log("📩 Received from server:", data);
        
        if (data.prediction) {
            // Update the display with the chord name
            predictionText.textContent = data.prediction;
            
            // Play the chord sound
            playChordSound(data.prediction);
            
            // Visual feedback for detected chords
            predictionText.classList.add('detected');
            setTimeout(() => {
                predictionText.classList.remove('detected');
            }, 1000);
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
            startFrameSending();
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

        // Draw the video frame to canvas - apply flip if needed
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        context.setTransform(1, 0, 0, 1, 0, 0); // Reset transform

        const base64Image = canvas.toDataURL('image/jpeg', 0.8);

        ws.send(JSON.stringify({ frame: base64Image }));
    }, 300); // Increased to 300ms to reduce server load and processing frequency
}