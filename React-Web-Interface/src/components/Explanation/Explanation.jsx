import './Explanation.css';

const Explanation = () => {
  return (
    <section className="explanation">
      <div className="container">
        <h2>How Our AI System Works</h2>
        
        <div className="explanation-content">
          <div className="explanation-step">
            <div className="step-number">1</div>
            <h3>Neural Architecture Search (NAS)</h3>
            <p>
              Instead of manually designing the neural network, our system uses NAS to automatically 
              explore thousands of possible architectures. It tests different configurations of:
            </p>
            <ul>
              <li>Number of convolutional blocks</li>
              <li>Filter multipliers</li>
              <li>Kernel sizes</li>
              <li>Batch normalization usage</li>
              <li>Dropout rates</li>
              <li>Dense layer configurations</li>
            </ul>
          </div>

          <div className="explanation-step">
            <div className="step-number">2</div>
            <h3>Automated Model Training</h3>
            <p>
              Each candidate architecture is trained on our dataset of lung and colon tissue images.
              The system evaluates performance and selects the best architecture based on validation accuracy.
            </p>
          </div>

          <div className="explanation-step">
            <div className="step-number">3</div>
            <h3>Superior Performance</h3>
            <p>
              The NAS-optimized model achieves 96.3% validation accuracy, significantly outperforming
              traditional manually-designed models (89.6% accuracy). It also shows better generalization
              with less overfitting.
            </p>
          </div>

          <div className="explanation-step">
            <div className="step-number">4</div>
            <h3>Image Analysis</h3>
            <p>
              When you upload an image, our model analyzes it through multiple convolutional layers
              to extract features, followed by classification layers that identify the tissue type
              with high confidence.
            </p>
          </div>
        </div>

        <div className="technical-details">
          <h3>Technical Details</h3>
          <div className="details-grid">
            <div className="detail-item">
              <h4>Dataset</h4>
              <p>LC25000 dataset with 25,000 images across 5 classes</p>
            </div>
            <div className="detail-item">
              <h4>Classes</h4>
              <p>Lung benign, Lung adenocarcinoma, Lung squamous cell carcinoma, Colon adenocarcinoma, Colon benign</p>
            </div>
            <div className="detail-item">
              <h4>Image Size</h4>
              <p>128×128 pixels</p>
            </div>
            <div className="detail-item">
              <h4>Training</h4>
              <p>15 epochs for NAS search, 30 epochs for final training</p>
            </div>
            <div className="detail-item">
              <h4>Framework</h4>
              <p>PyTorch with custom NAS implementation</p>
            </div>
            <div className="detail-item">
              <h4>Performance</h4>
              <p>96.3% accuracy, 0.056 validation loss</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Explanation;