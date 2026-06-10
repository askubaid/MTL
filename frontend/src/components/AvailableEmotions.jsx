import React, { useState, useEffect } from 'react';
import emotionCsv from '../emotion_labels.csv?raw';

const AvailableEmotions = () => {
  const [labels, setLabels] = useState([]);

  useEffect(() => {
    const lines = emotionCsv.trim().split('\n');
    const parsedLabels = lines.slice(1).map(line => line.split(',')[1].trim());
    setLabels(parsedLabels);
  }, []);

  if (!labels || labels.length === 0) return null;

  return (
    <div className="labels-column">
      <h3>Available Emotions</h3>
      <div className="dataset-meta" style={{ fontSize: '0.85em', color: '#666', marginBottom: '15px' }}>
        <p><strong>Dataset Name:</strong> DAIR-AI / Emotion</p>
        <p><strong>Dataset Size:</strong> {labels.length} classes (20,000 Twitter samples)</p>
        <p><strong>Creator:</strong> DAIR.AI</p>
      </div>
      <div className="labels-list">
        {labels.map((emotion, idx) => (
          <span key={idx} className="label-badge emotion-badge">{emotion.replace(/_/g, ' ')}</span>
        ))}
      </div>
    </div>
  );
};

export default AvailableEmotions;
