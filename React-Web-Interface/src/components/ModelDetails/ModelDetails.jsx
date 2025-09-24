import './ModelDetails.css';

const ModelDetails = () => {
  return (
    <section className="model-details">
      <div className="container">
        <h2>Model Architecture & Training Details</h2>
        
        <div className="architecture">
          <h3>NAS-Optimized Architecture</h3>
          <p>
            Our best model was discovered through Neural Architecture Search with the following configuration:
          </p>
          
          <div className="architecture-details">
            <div className="arch-section">
              <h4>Convolutional Blocks</h4>
              <ul>
                <li><strong>Number of blocks:</strong> 4</li>
                <li><strong>Filter multipliers:</strong> 2× base filters</li>
                <li><strong>Kernel sizes:</strong> [3, 5, 5, 3]</li>
                <li><strong>Batch normalization:</strong> Enabled</li>
                <li><strong>Convolutional dropout:</strong> 0.2</li>
              </ul>
            </div>
            
            <div className="arch-section">
              <h4>Classifier</h4>
              <ul>
                <li><strong>Dense units:</strong> [512, 256]</li>
                <li><strong>Dense dropout:</strong> 0.4</li>
                <li><strong>Output units:</strong> 5 (one per class)</li>
              </ul>
            </div>
            
            <div className="arch-section">
              <h4>Training Parameters</h4>
              <ul>
                <li><strong>Learning rate:</strong> 0.0001</li>
                <li><strong>Optimizer:</strong> Adam with weight decay</li>
                <li><strong>Loss function:</strong> Cross Entropy</li>
                <li><strong>Batch size:</strong> 32</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="training-process">
          <h3>Training Process</h3>
          <p>
            The model was trained using a two-phase approach:
          </p>
          
          <div className="phase">
            <h4>Phase 1: Neural Architecture Search</h4>
            <ul>
              <li>15 different configurations tested</li>
              <li>15 epochs per configuration</li>
              <li>Validation accuracy used to rank models</li>
              <li>Best configuration selected for final training</li>
            </ul>
          </div>
          
          <div className="phase">
            <h4>Phase 2: Final Training</h4>
            <ul>
              <li>Best configuration trained for 30 epochs</li>
              <li>ReduceLROnPlateau scheduler used</li>
              <li>Model checkpointing for best validation accuracy</li>
              <li>Final validation accuracy: 96.3%</li>
            </ul>
          </div>
        </div>
        
        <div className="performance">
          <h3>Performance Metrics</h3>
          <div className="metrics-grid">
            <div className="metric">
              <h4>96.3%</h4>
              <p>Validation Accuracy</p>
            </div>
            <div className="metric">
              <h4>0.056</h4>
              <p>Validation Loss</p>
            </div>
            <div className="metric">
              <h4>3.4%</h4>
              <p>Generalization Gap</p>
            </div>
            <div className="metric">
              <h4>+12.9%</h4>
              <p>Improvement on Hard Cases</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelDetails;