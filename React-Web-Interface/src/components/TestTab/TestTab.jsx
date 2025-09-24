import UploadInterface from '../UploadInterface/UploadInterface';
import './TestTab.css';

const TestTab = () => {
  return (
    <div className="test-tab">
      <div className="test-tab-header">
        <div className="container">
          <h2>AI Model Testing Interface</h2>
          <p>Upload tissue images to test our NAS-optimized cancer detection model with Explainable AI</p>
          <div className="model-badges">
            <span className="badge">99.04% Validation Accuracy</span>
            <span className="badge">Explainable AI</span>
            <span className="badge">Grad-CAM Visualizations</span>
          </div>
        </div>
      </div>
      <UploadInterface />
    </div>
  );
};

export default TestTab;