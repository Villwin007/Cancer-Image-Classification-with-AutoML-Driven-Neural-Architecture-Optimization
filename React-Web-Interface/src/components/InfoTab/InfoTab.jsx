import Hero from '../Hero/Hero';
import Explanation from '../Explanation/Explanation';
import ModelDetails from '../ModelDetails/ModelDetails';
import './InfoTab.css';

const InfoTab = () => {
  return (
    <div className="info-tab">
      <Hero />
      <Explanation />
      <ModelDetails />
    </div>
  );
};

export default InfoTab;