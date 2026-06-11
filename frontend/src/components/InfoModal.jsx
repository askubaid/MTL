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
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            components={{
              img: ({node, ...props}) => {
                let src = props.src;
                if (src && !src.startsWith('http')) {
                  // Resolve relative paths to the GitHub raw content
                  src = `https://raw.githubusercontent.com/askubaid/MTL/main/${src}`;
                }
                return <img {...props} src={src} style={{ maxWidth: '100%', height: 'auto' }} />;
              }
            }}
          >
            {readmeText}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default InfoModal;
