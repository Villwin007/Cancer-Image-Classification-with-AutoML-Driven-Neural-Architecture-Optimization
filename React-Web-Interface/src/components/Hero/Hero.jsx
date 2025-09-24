import './Hero.css';

const Hero = () => {
  return (
    <section className="hero">
      <div className="container">
        <div className="hero-content">
          <h2>Advanced Cancer Detection with Neural Architecture Search</h2>
          <p>
            Our system uses cutting-edge AutoML technology to automatically design 
            and train highly accurate convolutional neural networks for detecting 
            lung and colon cancer from tissue images.
          </p>
          <div className="hero-stats">
            <div className="stat">
              <h3>96.3%</h3>
              <p>Validation Accuracy</p>
            </div>
            <div className="stat">
              <h3>5</h3>
              <p>Cancer Types Detected</p>
            </div>
            <div className="stat">
              <h3>NAS</h3>
              <p>Neural Architecture Search</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;