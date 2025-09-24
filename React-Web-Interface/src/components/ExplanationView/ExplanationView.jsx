import './ExplanationView.css';

const ExplanationView = ({ explanation, diagnosis, confidence }) => {
  if (!explanation) return null;

  return (
    <div className="explanation-view">
      <h3>AI Explanation</h3>
      
      <div className="explanation-section">
        <h4>Why this diagnosis?</h4>
        <p className="explanation-text">{explanation.textual}</p>
      </div>

      {explanation.heatmap_image && (
        <div className="explanation-section">
          <h4>Key Areas Identified</h4>
          <p>Red areas show regions most influential in the diagnosis:</p>
          <div className="heatmap-container">
            <img 
              src={`data:image/png;base64,${explanation.heatmap_image}`} 
              alt="AI attention heatmap"
              className="heatmap-image"
            />
            <div className="heatmap-legend">
              <div className="legend-item">
                <span className="legend-color high-importance"></span>
                <span>High importance</span>
              </div>
              <div className="legend-item">
                <span className="legend-color medium-importance"></span>
                <span>Medium importance</span>
              </div>
              <div className="legend-item">
                <span className="legend-color low-importance"></span>
                <span>Low importance</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {explanation.key_factors && (
        <div className="explanation-section">
          <h4>Key Factors Considered</h4>
          <div className="factors-list">
            {explanation.key_factors.map((factor, index) => (
              <div key={index} className="factor-item">
                <span className="factor-icon">🔍</span>
                <span>{factor}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="confidence-breakdown">
        <h4>Confidence Analysis</h4>
        <div className="confidence-level">
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ width: `${confidence * 100}%` }}
            ></div>
          </div>
          <div className="confidence-labels">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
            <span>Very High</span>
          </div>
        </div>
        <p className="confidence-description">
          {confidence > 0.9 
            ? "The model is very confident in this diagnosis based on clear pathological features."
            : confidence > 0.7
            ? "Good confidence level with identifiable characteristic features."
            : "Moderate confidence. Additional clinical correlation may be beneficial."}
        </p>
      </div>
    </div>
  );
};

export default ExplanationView;