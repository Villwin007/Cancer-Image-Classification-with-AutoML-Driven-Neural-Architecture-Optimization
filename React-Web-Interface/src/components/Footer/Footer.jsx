// import './Footer.css';

// const Footer = () => {
//   return (
//     <footer className="footer">
//       <div className="container">
//         <div className="footer-content">
//           <div className="footer-section">
//             <h3>Lung & Colon Cancer Detection</h3>
//             <p>Advanced AI-powered medical image analysis using Neural Architecture Search.</p>
//           </div>
          
//           <div className="footer-section">
//             <h4>Research</h4>
//             <p>This project is based on cutting-edge research in AutoML and medical AI.</p>
//           </div>
          
//           <div className="footer-section">
//             <h4>Important Notice</h4>
//             <p>This tool is for research purposes only and should not be used for actual medical diagnosis.</p>
//           </div>

//         </div>
        
//         <div className="footer-bottom">
//           <p>&copy; 2025 Cancer Detection Research Project. All rights reserved.</p>
//         </div>
//       </div>
//     </footer>
//   );
// };

// export default Footer;

import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h3>Cancer Image Classification with AutoML-Driven 
Neural Architecture Optimization</h3>
            <p>Advanced AI-powered medical image analysis using Neural Architecture Search and Explainable AI.</p>
            {/* <div className="project-stats">
              <div className="stat">
                <strong>99.04%</strong>
                <span>Validation Accuracy</span>
              </div>
              <div className="stat">
                <strong>5</strong>
                <span>Cancer Types</span>
              </div>
              <div className="stat">
                <strong>NAS</strong>
                <span>Optimized</span>
              </div>
            </div> */}
          </div>
          
          <div className="footer-section">
            <h4>Developer Information</h4>
            <div className="developer-info">
              <div className="info-item">
                <strong>Project:</strong> Lung & Colon Cancer Detection System
              </div>
              <div className="info-item">
                <strong>Developer:</strong> Dhanush Saravanan
              </div>
              <div className="info-item">
                <strong>Email:</strong> s.dhanush1106@gmail.com
              </div>
              <div className="info-item">
                <strong>Institution:</strong> Kalasalingam Academy of Research and Education
              </div>
              <div className="info-item">
                <strong>Project Type:</strong> Capstone Project / Research
              </div>
            </div>
          </div>
          
          <div className="footer-section">
            <h4>Technical Details</h4>
            <div className="tech-stack">
              <div className="tech-item">
                <span className="tech-icon">🧠</span>
                <span>PyTorch + Neural Architecture Search</span>
              </div>
              <div className="tech-item">
                <span className="tech-icon">⚛️</span>
                <span>React.js + Vite Frontend</span>
              </div>
              <div className="tech-item">
                <span className="tech-icon">🐍</span>
                <span>Flask Backend API</span>
              </div>
              <div className="tech-item">
                <span className="tech-icon">🔍</span>
                <span>Grad-CAM Explainable AI</span>
              </div>
            </div>
          </div>
          
          <div className="footer-section">
            <h4>Research & Acknowledgments</h4>
            <p>This project demonstrates the power of AutoML in medical image analysis, achieving state-of-the-art results through Neural Architecture Search.</p>
            <div className="acknowledgments">
              <div className="ack-item">
                <strong>Dataset: </strong> LC25000 Lung & Colon Cancer
              </div>
              <div className="ack-item">
                <strong>Validation:</strong> 99.04% Accuracy
              </div>
              <div className="ack-item">
                <strong>Features:</strong> Explainable AI with Heatmaps
              </div>
            </div>
          </div>
        </div>
        
        <div className="footer-bottom">
          <div className="footer-bottom-content">
            <div className="copyright">
              <p>&copy; 2025 Cancer Image Classification with AutoML-Driven 
Neural Architecture Optimization. Developed as part of Capstone Project.</p>
              <p>This research demonstrates advanced AI applications in medical diagnostics.</p>
            </div>
            <div className="footer-links">
              <div className="link-group">
                <strong>Technology Stack: </strong>
                <span>PyTorch - </span>
                <span>React - </span>
                <span>Flask - </span>
                <span>Neural Architecture Search</span>
              </div>
              <div className="link-group">
                <strong>Research Areas: </strong>
                <span>Medical AI - </span>
                <span>Explainable AI - </span>
                <span>AutoML - </span>
                <span>Computer Vision</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;