import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import 'github-markdown-css/github-markdown.css';
import readmeText from '../../../README.md?raw';

const InfoModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '900px', width: '95%', maxHeight: '90vh', overflowY: 'auto' }}>
        <button className="modal-close" onClick={onClose} style={{ position: 'sticky', top: '10px', float: 'right' }}>&times;</button>
        <div className="markdown-body" style={{ padding: '20px', textAlign: 'left', borderRadius: '8px' }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {readmeText}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default InfoModal;
