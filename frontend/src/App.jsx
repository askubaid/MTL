import React, { useState, useEffect } from 'react';
import './index.css';
import PredictionCard from './components/PredictionCard';
import InfoModal from './components/InfoModal';

function App() {
  const [text, setText] = useState('');
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [labels, setLabels] = useState({ intents: [], emotions: [] });
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const fetchLabels = async () => {
      try {
        const response = await fetch(`${backendUrl}/labels`);
        if (response.ok) {
          const data = await response.json();
          setLabels(data);
        }
      } catch (err) {
        console.error("Failed to fetch labels", err);
      }
    };
    fetchLabels();
  }, [backendUrl]);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error(err);
      setError('An error occurred while connecting to the backend. Is it running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <button className="btn-info" onClick={() => setIsModalOpen(true)}>
        InfoGraphics
      </button>
      <InfoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      
      <header className="header">
        <h1>Multi-Task Learning UI</h1>
        <p>Intent & Emotion Classification</p>
      </header>

      
      <main>
        <section className="input-section">
          <div className="backend-url-input" style={{ marginBottom: '15px' }}>
            <label htmlFor="backendUrl" style={{ marginRight: '10px', fontWeight: 'bold' }}>Backend URL:</label>
            <input 
              id="backendUrl"
              type="text" 
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              placeholder="https://your-ngrok-url.ngrok.io"
              style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #ccc', marginTop: '5px' }}
            />
          </div>
          <textarea 
            placeholder="Type your review or text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button 
            className="btn-submit" 
            onClick={handleSubmit} 
            disabled={loading || !text.trim()}
          >
            {loading ? <span className="loading-spinner"></span> : 'Analyze'}
          </button>
          {error && <p style={{color: 'red', marginTop: '10px'}}>{error}</p>}
        </section>

        {prediction && (
          <section className="prediction-container">
            <PredictionCard 
              title="Predicted Intent"
              value={prediction.intent}
              confidence={prediction.intent_confidence}
            />
            <PredictionCard 
              title="Predicted Emotion"
              value={prediction.emotion}
              confidence={prediction.emotion_confidence}
            />
          </section>
        )}

        {(labels.intents.length > 0 || labels.emotions.length > 0) && (
          <section className="labels-section">
            <div className="labels-column">
              <h3>Available Intents</h3>
              <div className="labels-list">
                {labels.intents.map((intent, idx) => (
                  <span key={idx} className="label-badge">{intent.replace(/_/g, ' ')}</span>
                ))}
              </div>
            </div>
            <div className="labels-column">
              <h3>Available Emotions</h3>
              <div className="labels-list">
                {labels.emotions.map((emotion, idx) => (
                  <span key={idx} className="label-badge emotion-badge">{emotion.replace(/_/g, ' ')}</span>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
