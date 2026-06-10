import React, { useEffect, useState } from 'react';

const InfoModal = ({ isOpen, onClose }) => {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetch('http://localhost:8000/evaluation-info')
        .then(res => res.json())
        .then(data => {
          setInfo(data);
          setLoading(false);
        })
        .catch(err => {
          console.error(err);
          setLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>&times;</button>
        <h2>Model InfoGraphics & Evaluation</h2>

        {loading ? (
          <div className="modal-loading"><span className="loading-spinner"></span> Loading...</div>
        ) : info ? (
          <div className="modal-body">
            <section className="info-section">
              <h3>Training Configuration</h3>
              <ul>
                <li><strong>Epochs:</strong> {info.epochs}</li>
                <li><strong>Batch Size:</strong> {info.hyperparameters.batch_size}</li>
                <li><strong>Learning Rate:</strong> {info.hyperparameters.learning_rate}</li>
                <li><strong>Max Length:</strong> {info.hyperparameters.max_length}</li>
              </ul>
            </section>

            <section className="info-section">
              <h3>Datasets</h3>
              <ul>
                <li> 1) CLINCS-150 Dataset (Intent Detection) - 23,700 samples </li>
                <li> 2) Go_Emotions Dataset (Emotion Detection) - 58,000 samples</li>
              </ul>
            </section>

            <section className="info-section">
              <h3>Evaluation Summary</h3>
              {info.summary && info.summary.length > 0 && (
                <div className="eval-stats">
                  <div className="stat-box">
                    <span>Intent Accuracy</span>
                    <strong>{(parseFloat(info.summary[0].intent_accuracy) * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="stat-box">
                    <span>Intent F1</span>
                    <strong>{info.summary[0].intent_macro_f1}</strong>
                  </div>
                  <div className="stat-box">
                    <span>Emotion Accuracy</span>
                    <strong>{(parseFloat(info.summary[0].emotion_accuracy) * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="stat-box">
                    <span>Emotion F1</span>
                    <strong>{info.summary[0].emotion_macro_f1}</strong>
                  </div>
                </div>
              )}
            </section>

            <section className="info-section">
              <h3>Learning Curves</h3>
              <img
                src="http://localhost:8000/evaluation/learning_curves.png"
                alt="Learning Curves"
                className="learning-curves-img"
              />
            </section>

            <section className="info-section">
              <h3>Detailed Evaluation Report</h3>
              <pre className="eval-details-pre">{info.details}</pre>
            </section>
          </div>
        ) : (
          <p>Failed to load data.</p>
        )}
      </div>
    </div>
  );
};

export default InfoModal;
