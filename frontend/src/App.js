import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// This is the backend URL for Render.  
const API_BASE_URL = 'https://youtube-to-text-51uc.onrender.com';

function App() {
const [youtubeUrl, setYoutubeUrl] = useState('');
const [loading, setLoading] = useState(false);
const [result, setResult] = useState(null);
const [error, setError] = useState('');
const [conversionCount, setConversionCount] = useState(0);
const [timeRemaining, setTimeRemaining] = useState(0);
const [showFeedbackModal, setShowFeedbackModal] = useState(false);
const [feedbackForm, setFeedbackForm] = useState({
  name: '',
  email: '',
  comments: ''
});
const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
const [feedbackSuccess, setFeedbackSuccess] = useState(false);

// Fetch conversion count on component mount
useEffect(() => {
  const fetchConversionCount = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversion-count`);
      setConversionCount(response.data.count);
    } catch (err) {
      console.error('Failed to fetch conversion count');
    }
  };
  fetchConversionCount();
}, []);

// Timer effect for countdown
useEffect(() => {
  let timer;
  if (loading && timeRemaining > 0) {
    timer = setInterval(() => {
      setTimeRemaining(prev => prev - 1);
    }, 1000);
  }
  return () => clearInterval(timer);
}, [loading, timeRemaining]);

// Function to estimate processing time based on URL length
const estimateProcessingTimeFromURL = (url) => {
  const baseTime = 180; // 3 minutes base
  const urlLength = url.length;
  
  // Add time based on URL length as a rough proxy
  const additionalTime = Math.min(120, Math.floor(urlLength / 10)); // Max 2 additional minutes
  
  return baseTime + additionalTime; // Total: 3-5 minutes
};

// Function to estimate processing time based on transcript length
const estimateProcessingTimeFromTranscript = (transcriptLength) => {
  // 1 second per 100 characters, minimum 1 minute
  return Math.max(60, Math.floor(transcriptLength / 100));
};

// Format time as MM:SS
const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
};

const handleSubmit = async (e) => {
  e.preventDefault();
  setLoading(true);
  setError('');
  setResult(null);
  
  // Start timer with URL-based estimate
  const initialEstimate = estimateProcessingTimeFromURL(youtubeUrl);
  setTimeRemaining(initialEstimate);

  try {
    // Step 1: Get transcript info for accurate timing
    const transcriptInfoPromise = axios.post(`${API_BASE_URL}/get-transcript-info`, {
      youtube_url: youtubeUrl
    });
    
    // Step 2: Start main processing
    const processingPromise = axios.post(`${API_BASE_URL}/clean-transcript`, {
      youtube_url: youtubeUrl
    });
    
    // Wait for transcript info and update timer
    try {
      const transcriptInfo = await transcriptInfoPromise;
      const accurateEstimate = transcriptInfo.data.estimated_processing_time;
      console.log('Transcript info received:', transcriptInfo.data);
      console.log('Updating timer from URL estimate to accurate estimate:', accurateEstimate);
      setTimeRemaining(accurateEstimate);
    } catch (infoError) {
      console.log('Could not get transcript info, continuing with URL estimate:', infoError);
    }
    
    // Wait for main processing to complete
    const response = await processingPromise;
    
    setResult(response.data);
    // Increment counter after successful conversion
    setConversionCount(prev => prev + 1);
    
  } catch (err) {
    setError(err.response?.data?.detail || 'An error occurred');
  } finally {
    setLoading(false);
    setTimeRemaining(0);
  }
};

const handleFeedbackSubmit = async (e) => {
  e.preventDefault();
  if (!feedbackForm.comments.trim()) {
    alert('Please enter your comments');
    return;
  }

  setFeedbackSubmitting(true);
  try {
    await axios.post(`${API_BASE_URL}/send-feedback`, feedbackForm);
    setFeedbackSuccess(true);
    setFeedbackForm({ name: '', email: '', comments: '' });
    setTimeout(() => {
      setShowFeedbackModal(false);
      setFeedbackSuccess(false);
    }, 2000);
  } catch (err) {
    alert('Failed to send feedback. Please try again.');
  } finally {
    setFeedbackSubmitting(false);
  }
};

const closeFeedbackModal = () => {
  setShowFeedbackModal(false);
  setFeedbackForm({ name: '', email: '', comments: '' });
  setFeedbackSuccess(false);
};

const copyToClipboard = (text) => {
  navigator.clipboard.writeText(text);
  alert('Copied to clipboard!');
};

const downloadHTML = (text, filename) => {
  const currentDate = new Date().toLocaleDateString();
  const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YouTube Video Transcript</title>
  <style>
      body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          max-width: 800px;
          margin: 0 auto;
          padding: 40px 20px;
          line-height: 1.8;
          color: #333;
          background-color: #fff;
      }
      .header {
          border-bottom: 2px solid #e0e0e0;
          margin-bottom: 30px;
          padding-bottom: 20px;
      }
      h1 {
          color: #444;
          margin-bottom: 10px;
          font-size: 2rem;
      }
      .meta {
          color: #666;
          font-size: 0.9rem;
          margin-bottom: 20px;
      }
      .transcript {
          font-size: 1.1rem;
          line-height: 2;
      }
      p {
          margin-bottom: 1.2em;
      }
      @media print {
          body { margin: 20px; }
      }
  </style>
</head>
<body>
  <div class="header">
      <h1>YouTube Video Transcript</h1>
      <div class="meta">
          <strong>Generated:</strong> ${currentDate}<br>
          <strong>Source:</strong> ${youtubeUrl}
      </div>
  </div>
  
  <div class="transcript">
      ${text.replace(/\n/g, '</p><p>').replace(/^/, '<p>').replace(/$/, '</p>')}
  </div>
</body>
</html>`;
  
  const blob = new Blob([htmlContent], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

const downloadMarkdown = (text, filename) => {
  const currentDate = new Date().toLocaleDateString();
  const markdownContent = `# YouTube Video Transcript

**Generated:** ${currentDate}  
**Source:** ${youtubeUrl}

---

${text.replace(/\n/g, '\n\n')}`;
  
  const blob = new Blob([markdownContent], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

return (
  <div className="App">
    <header className="App-header">
      <h1>Youtube to Text Generator</h1>
      <p>AI-powered video transcription tool</p>
      <div className="conversion-counter">
        <span className="counter-label">Videos Processed:</span>
        <span className="counter-number">{conversionCount.toLocaleString()}</span>
      </div>
    </header>

    <main className="main-content">
      <form onSubmit={handleSubmit} className="main-form">
        <div className="url-section">
          <label htmlFor="youtube-url">YouTube URL:</label>
          <input
            id="youtube-url"
            type="url"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="Paste YouTube URL here..."
            required
            className="url-input"
          />
        </div>

        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? (
            <>
              <span className="loading-spinner"></span>
              Processing...
              {timeRemaining > 0 && (
                <div style={{ marginTop: '10px', fontSize: '0.9rem' }}>
                  Estimated time remaining: {formatTime(timeRemaining)}
                </div>
              )}
            </>
          ) : (
            '✨ Generate Text'
          )}
        </button>
      </form>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {result && (
        <div className="results">
          <div className="transcript-section">
            <h3>Generated Transcript</h3>
            <div className="transcript-box cleaned">
              <div dangerouslySetInnerHTML={{ 
                __html: result.cleaned_text.replace(/\n/g, '<br>') 
              }} />
              <div className="action-buttons">
                <button onClick={() => copyToClipboard(result.cleaned_text)}>
                  📋 Copy
                </button>
                <button onClick={() => downloadHTML(result.cleaned_text, 'transcript.html')}>
                  💾 Download HTML
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>

    {/* Feedback Modal */}
    {showFeedbackModal && (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1000
      }}>
        <div style={{
          backgroundColor: 'white',
          padding: '30px',
          borderRadius: '10px',
          width: '90%',
          maxWidth: '500px',
          maxHeight: '80%',
          overflow: 'auto'
        }}>
          {feedbackSuccess ? (
            <div style={{ textAlign: 'center' }}>
              <h3 style={{ color: '#28a745', marginBottom: '10px' }}>✅ Thank you!</h3>
              <p>Your feedback has been sent successfully.</p>
            </div>
          ) : (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3 style={{ margin: 0 }}>Feedback</h3>
                <button 
                  onClick={closeFeedbackModal}
                  style={{
                    background: 'none',
                    border: 'none',
                    fontSize: '24px',
                    cursor: 'pointer',
                    color: '#666'
                  }}
                >
                  ×
                </button>
              </div>
              
              <p style={{ marginBottom: '20px', color: '#666' }}>
                Found a bug or issue? Or want to leave a comment? Report it here.
              </p>

              <form onSubmit={handleFeedbackSubmit}>
                <div style={{ marginBottom: '15px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', color: '#333' }}>
                    Name (optional):
                  </label>
                  <input
                    type="text"
                    value={feedbackForm.name}
                    onChange={(e) => setFeedbackForm(prev => ({ ...prev, name: e.target.value }))}
                    style={{
                      width: '100%',
                      padding: '10px',
                      border: '1px solid #ddd',
                      borderRadius: '5px',
                      fontSize: '14px'
                    }}
                  />
                </div>

                <div style={{ marginBottom: '15px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', color: '#333' }}>
                    Email (optional):
                  </label>
                  <input
                    type="email"
                    value={feedbackForm.email}
                    onChange={(e) => setFeedbackForm(prev => ({ ...prev, email: e.target.value }))}
                    style={{
                      width: '100%',
                      padding: '10px',
                      border: '1px solid #ddd',
                      borderRadius: '5px',
                      fontSize: '14px'
                    }}
                  />
                </div>

                <div style={{ marginBottom: '20px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', color: '#333' }}>
                    Comments: *
                  </label>
                  <textarea
                    value={feedbackForm.comments}
                    onChange={(e) => setFeedbackForm(prev => ({ ...prev, comments: e.target.value }))}
                    required
                    rows={5}
                    style={{
                      width: '100%',
                      padding: '10px',
                      border: '1px solid #ddd',
                      borderRadius: '5px',
                      fontSize: '14px',
                      resize: 'vertical'
                    }}
                    placeholder="Please describe any bugs, issues, or feedback..."
                  />
                </div>

                <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                  <button
                    type="button"
                    onClick={closeFeedbackModal}
                    style={{
                      padding: '10px 20px',
                      border: '1px solid #ddd',
                      borderRadius: '5px',
                      background: '#f8f9fa',
                      cursor: 'pointer'
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={feedbackSubmitting}
                    style={{
                      padding: '10px 20px',
                      border: 'none',
                      borderRadius: '5px',
                      background: '#007bff',
                      color: 'white',
                      cursor: feedbackSubmitting ? 'not-allowed' : 'pointer',
                      opacity: feedbackSubmitting ? 0.7 : 1
                    }}
                  >
                    {feedbackSubmitting ? 'Sending...' : 'Send Feedback'}
                  </button>
                </div>
              </form>
            </>
          )}
        </div>
      </div>
    )}

    {/* Feedback Link */}
    <div style={{
      textAlign: 'center',
      marginTop: '40px',
      paddingTop: '20px',
      borderTop: '1px solid #e0e0e0'
    }}>
      <button
        onClick={() => setShowFeedbackModal(true)}
        style={{
          background: 'none',
          border: 'none',
          color: '#007bff',
          textDecoration: 'underline',
          cursor: 'pointer',
          fontSize: '14px'
        }}
      >
        Feedback
      </button>
    </div>
  </div>
);
}

export default App;
