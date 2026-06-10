import React, { useEffect, useState } from 'react';

const PredictionCard = ({ title, value, confidence }) => {
  const [animWidth, setAnimWidth] = useState(0);

  useEffect(() => {
    // Reset and animate the confidence bar
    setAnimWidth(0);
    const timer = setTimeout(() => {
      setAnimWidth(confidence * 100);
    }, 50);
    return () => clearTimeout(timer);
  }, [confidence]);

  return (
    <div className="prediction-card">
      <div className="prediction-title">{title}</div>
      <div className="prediction-value">{value.replace(/_/g, ' ')}</div>
      <div className="prediction-conf">
        Confidence: {(confidence * 100).toFixed(1)}%
      </div>
      <div className="conf-bar-container">
        <div 
          className="conf-bar" 
          style={{ width: `${animWidth}%` }} 
        />
      </div>
    </div>
  );
};

export default PredictionCard;
