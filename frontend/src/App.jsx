import React, { useState, useEffect } from 'react';
import './index.css';
import PredictionCard from './components/PredictionCard';
import InfoModal from './components/InfoModal';
import AvailableIntents from './components/AvailableIntents';
import AvailableEmotions from './components/AvailableEmotions';
import logoUrl from './assets/logo.png';

function App() {
  const [text, setText] = useState('');
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

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
          {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}
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

        <section className="labels-section">
          <AvailableIntents />
          <AvailableEmotions />
        </section>

      </main>

      <footer className="app-footer" style={{ marginTop: '40px', padding: '20px', borderTop: '1px solid #eee', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '30px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
        <img src={logoUrl} alt="IST Logo" style={{ width: '100px', height: 'auto' }} />
        <div style={{ textAlign: 'left', fontSize: '0.95em', color: '#444', lineHeight: '1.4' }}>
          <p style={{ margin: '0' }}><strong>Course:</strong> Advanced Generative Computing Systems</p>
          <p style={{ margin: '0' }}><strong>Submitted to:</strong> Dr. Benish Amin</p>
          <p style={{ margin: '0' }}><strong>Developed By:</strong> Ubaid Ur Rehman</p>
          <p style={{ margin: '0', color: '#666' }}>Institute of Space Technology Islamabad Pakistan</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
