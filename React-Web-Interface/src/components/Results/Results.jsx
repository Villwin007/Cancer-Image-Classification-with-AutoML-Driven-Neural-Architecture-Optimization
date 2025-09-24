import ExplanationView from '../ExplanationView/ExplanationView';
import './Results.css';

const Results = ({ results, isLoading, onNewAnalysis }) => {
  if (!results && !isLoading) return null;

  if (isLoading) {
    return (
      <section className="results">
        <div className="container">
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing image with NAS-optimized model...</p>
            <p className="loading-subtext">Generating detailed explanation...</p>
          </div>
        </div>
      </section>
    );
  }

  const { predictions, confidence, diagnosis, diagnosis_type, explanation } = results;

  return (
    <section className="results">
      <div className="container">
        <h2>AI Analysis Results</h2>
        <div className="model-badge">
          <span>NAS-Optimized Model • 99.04% Validation Accuracy • Explainable AI</span>
        </div>
        
        <div className="results-content">
          <div className="confidence-meter">
            <h3>Model Confidence: {results.confidence_percentage}</h3>
            <div className="meter-bar">
              <div 
                className="meter-fill" 
                style={{ width: `${confidence * 100}%` }}
              ></div>
            </div>
          </div>
          
          <div className="diagnosis">
            <h3>Diagnosis: {diagnosis}</h3>
            <p className={`diagnosis-text ${diagnosis_type === 'benign' ? 'benign' : 'cancerous'}`}>
              {diagnosis_type === 'benign' 
                ? 'The tissue appears to be non-cancerous.' 
                : 'The tissue shows characteristics of cancer. Please consult a medical professional.'}
            </p>
          </div>
          
          <div className="predictions">
            <h3>Detailed Probability Distribution</h3>
            <div className="prediction-bars">
              {predictions.map((pred, index) => (
                <div key={index} className="prediction-item">
                  <div className="prediction-label">
                    <span>{pred.class}</span>
                    <span>{pred.percentage}</span>
                  </div>
                  <div className="prediction-bar">
                    <div 
                      className="prediction-fill" 
                      style={{ width: `${pred.probability * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Explanation Section */}
          <ExplanationView 
            explanation={explanation}
            diagnosis={diagnosis}
            confidence={confidence}
          />
          
          <div className="results-actions">
            <button onClick={onNewAnalysis} className="new-analysis-btn">
              Analyze Another Image
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Results;