import React, { useState, useEffect } from 'react';
import intentCsv from '../intent_labels.csv?raw';

const AvailableIntents = () => {
  const [labels, setLabels] = useState([]);

  useEffect(() => {
    const lines = intentCsv.trim().split('\n');
    const parsedLabels = lines.slice(1).map(line => line.split(',')[1].trim());
    setLabels(parsedLabels);
  }, []);

  if (!labels || labels.length === 0) return null;

  return (
    <div className="labels-column">
      <h3>Available Intents</h3>
      <div className="dataset-meta" style={{ fontSize: '0.85em', color: '#666', marginBottom: '15px' }}>
        <p><strong>Dataset Name:</strong> CLINC150 / Customer Intents</p>
        <p><strong>Dataset Size:</strong> {labels.length} classes (22,500+ samples)</p>
        <p><strong>Creator:</strong> CLINC</p>
      </div>
      <div className="labels-list">
        {labels.map((intent, idx) => (
          <span key={idx} className="label-badge">{intent.replace(/_/g, ' ')}</span>
        ))}
      </div>
    </div>
  );
};

export default AvailableIntents;
